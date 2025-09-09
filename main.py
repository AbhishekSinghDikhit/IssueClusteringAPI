import logging
import asyncio
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core.exceptions import ResourceExhausted
from bson import ObjectId
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# -------------------
# Logging setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# -------------------
# MongoDB setup
# -------------------
try:
    MONGO_URI = os.getenv("MONGODB_URI")
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client["data"]
    issues_collection = db["issues"]
    clustered_collection = db["ClusteredIssues"]
    logger.info("✅ Successfully connected to MongoDB")
except Exception as e:
    logger.exception("❌ Failed to connect to MongoDB")
    raise e

# -------------------
# Gemini setup
# -------------------
gemini_client = None
rate_limiter = asyncio.Semaphore(10)  # Limit to 10 concurrent calls (RPM safeguard)

@app.on_event("startup")
async def startup_event():
    global gemini_client
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:]
        logger.info(f"🔑 Gemini API key loaded: {masked_key}")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai
    else:
        logger.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in .env")
        raise RuntimeError("Gemini API key not configured")

    # Start watching for new issues
    asyncio.create_task(watch_issues())

@app.on_event("shutdown")
async def shutdown_event():
    mongo_client.close()
    logger.info("🔌 MongoDB connection closed")

# -------------------
# Pydantic model for new issue
# -------------------
class Issue(BaseModel):
    type: str
    description: str
    location: dict
    status: str = "unresolved"

# -------------------
# Gemini summarizer
# -------------------
async def generate_summary(descriptions: list[str]) -> str:
    """Generate a robust, relevant summary using Gemini API."""
    async with rate_limiter:
        try:
            # Clean + deduplicate + trim long descriptions
            cleaned = list({d.strip()[:300] for d in descriptions if d and d.strip()})
            cleaned = cleaned[:10]  # cap at 10 items

            if not cleaned:
                return "No valid descriptions provided."

            prompt = (
                "Summarize the following civic issues into ONE short, clear, and "
                "explanatory description that highlights the main problem. "
                "Avoid repetition and make it understandable for municipal/government officials.\n\n"
                "Complaints:\n- " + "\n- ".join(cleaned)
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=400,
                )
            )

            if response and response.text:
                return response.text.strip()

            if response and getattr(response, "candidates", None):
                candidate = response.candidates[0]
                if candidate and candidate.content and candidate.content.parts:
                    parts = [p.text for p in candidate.content.parts if hasattr(p, "text")]
                    if parts:
                        return " ".join(parts).strip()

            logger.warning("⚠️ Gemini returned empty response, using fallback")
            return " | ".join(cleaned[:3])

        except ResourceExhausted:
            logger.error("❌ Gemini rate limit exceeded")
            await asyncio.sleep(1)
            return " | ".join(descriptions[:3]) if descriptions else "Rate limit exceeded."
        except Exception as e:
            logger.error(f"❌ Gemini summarization failed: {e}")
            return " | ".join(descriptions[:3]) if descriptions else "No description available."

# -------------------
# Utils
# -------------------
def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def to_python_types(doc):
    """Convert NumPy and other non-serializable data types to pure Python types."""
    if isinstance(doc, dict):
        return {k: to_python_types(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [to_python_types(v) for v in doc]
    elif isinstance(doc, (np.integer, np.int64, np.int32)):
        return int(doc)
    elif isinstance(doc, (np.floating, np.float64, np.float32)):
        return float(doc)
    elif isinstance(doc, np.ndarray):
        return doc.tolist()
    else:
        return doc

# -------------------
# Priority function
# -------------------
def calculate_priority(count: int) -> str:
    if count >= 10:
        return "Critical"
    elif count >= 5:
        return "High"
    elif count >= 3:
        return "Medium"
    else:
        return "Low"

# -------------------
# Distance calculation
# -------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth (in km)"""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------
# Process single issue
# -------------------
async def process_single_issue(issue):
    """Process a single issue and assign it to a cluster if applicable."""
    if "location" not in issue or not issue["location"]:
        logger.warning(f"❌ Issue {issue['_id']} has no location data")
        return None

    coords_str = issue["location"].get("coordinates")
    if not coords_str or not isinstance(coords_str, str):
        logger.warning(f"❌ Issue {issue['_id']} has invalid coordinates format")
        return None

    try:
        lat, lon = map(float, coords_str.split(","))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            logger.warning(f"❌ Coordinates out of range for issue {issue['_id']}: {coords_str}")
            return None
    except ValueError as e:
        logger.warning(f"❌ Invalid coordinates format for issue {issue['_id']}: {coords_str}, error: {e}")
        return None

    issue_type = issue.get("type")
    if not issue_type:
        logger.warning(f"❌ Issue {issue['_id']} has no type, skipping")
        return None

    issue_id = str(issue["_id"])
    issue_description = issue.get("description", "")
    distance_threshold = 0.5  # 500 meters

    # Check for existing clusters
    existing_clusters = await clustered_collection.find({"issue_type": issue_type}).to_list(1000)

    for cluster in existing_clusters:
        cluster_lat, cluster_lon = cluster["location_center"]
        distance = haversine_distance(lat, lon, cluster_lat, cluster_lon)

        if distance <= distance_threshold:
            # Add to existing cluster
            old_count = cluster["count"]
            new_count = old_count + 1
            new_center_lat = (cluster_lat * old_count + lat) / new_count
            new_center_lon = (cluster_lon * old_count + lon) / new_count

            all_issue_ids = cluster["issues"] + [issue_id]
            all_issues = await issues_collection.find(
                {"_id": {"$in": [ObjectId(i) for i in all_issue_ids]}}
            ).to_list(None)
            all_descriptions = [iss.get("description", "") for iss in all_issues]
            new_summary = await generate_summary(all_descriptions)
            new_priority = calculate_priority(new_count)

            await clustered_collection.update_one(
                {"_id": ObjectId(cluster["_id"])},
                {
                    "$push": {"issues": issue_id},
                    "$inc": {"count": 1},
                    "$set": {
                        "location_center": [new_center_lat, new_center_lon],
                        "general_description": new_summary,
                        "priority": new_priority,
                        "updatedAt": datetime.utcnow()
                    }
                }
            )
            logger.info(f"✅ Added issue {issue_id} to existing cluster {cluster['_id']} (issue_type: {issue_type})")
            return {"status": "added_to_cluster", "cluster_id": str(cluster["_id"])}

    # Check for other issues with same type and location
    similar_issues = await issues_collection.find({
        "status": {"$ne": "resolved"},
        "type": issue_type,
        "_id": {"$ne": ObjectId(issue_id)}
    }).to_list(1000)

    nearby_issues = []
    for other_issue in similar_issues:
        if "location" not in other_issue or not other_issue["location"]:
            continue
        other_coords = other_issue["location"].get("coordinates")
        if not other_coords or not isinstance(other_coords, str):
            continue
        try:
            other_lat, other_lon = map(float, other_coords.split(","))
            if not (-90 <= other_lat <= 90 and -180 <= other_lon <= 180):
                continue
            distance = haversine_distance(lat, lon, other_lat, other_lon)
            if distance <= distance_threshold:
                nearby_issues.append(other_issue)
        except ValueError:
            continue

    if len(nearby_issues) >= 1:  # Need at least one other issue to form a cluster
        new_cluster_id = await clustered_collection.count_documents({}) + 1
        issue_ids = [issue_id] + [str(ni["_id"]) for ni in nearby_issues]
        descriptions = [issue_description] + [ni.get("description", "") for ni in nearby_issues]
        coords = [[lat, lon]] + [[float(c.split(",")[0]), float(c.split(",")[1])] for c in [ni["location"]["coordinates"] for ni in nearby_issues]]
        center_lat = np.mean([c[0] for c in coords])
        center_lon = np.mean([c[1] for c in coords])

        cluster_doc = {
            "cluster_id": new_cluster_id,
            "issue_type": issue_type,
            "issues": issue_ids,
            "location_center": [float(center_lat), float(center_lon)],
            "count": len(issue_ids),
            "general_description": await generate_summary(descriptions),
            "priority": calculate_priority(len(issue_ids)),
            "createdAt": datetime.utcnow()
        }

        cluster_doc = to_python_types(cluster_doc)
        result = await clustered_collection.insert_one(cluster_doc)
        logger.info(f"✅ Created new cluster {result.inserted_id} with {len(issue_ids)} issues (issue_type: {issue_type})")
        return {"status": "new_cluster", "cluster_id": str(result.inserted_id)}

    logger.info(f"ℹ️ Issue {issue_id} not clustered: no nearby issues of type {issue_type}")
    return None

# -------------------
# Watch issues collection
# -------------------
async def watch_issues():
    """Watch for new issues and process them automatically."""
    try:
        async with issues_collection.watch([{
            "$match": {"operationType": {"$in": ["insert"]}}
        }]) as stream:
            async for change in stream:
                if change["operationType"] == "insert":
                    issue = change["fullDocument"]
                    logger.info(f"📥 Detected new issue {issue['_id']}")
                    await process_single_issue(issue)
    except Exception as e:
        logger.exception(f"❌ Error in watch_issues: {e}")

# -------------------
# Add issue endpoint
# -------------------
@app.post("/add-issue")
async def add_issue(issue: Issue):
    try:
        issue_doc = issue.dict()
        issue_doc["createdAt"] = datetime.utcnow()
        result = await issues_collection.insert_one(issue_doc)
        issue_doc["_id"] = str(result.inserted_id)
        logger.info(f"✅ Inserted new issue {issue_doc['_id']}")
        # Process the issue for clustering
        cluster_result = await process_single_issue(issue_doc)
        return {"status": "issue_added", "issue_id": issue_doc["_id"], "cluster_result": cluster_result}
    except Exception as e:
        logger.exception(f"❌ Error adding issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------
# Clustering endpoint
# -------------------
@app.get("/run-clustering")
async def run_clustering():
    try:
        logger.info("🚀 Clustering process started...")

        # Fetch unresolved issues not in clusters
        existing_cluster_issues = await clustered_collection.distinct("issues")
        existing_issue_ids = [ObjectId(issue_id) for issue_id in existing_cluster_issues]
        
        issues = await issues_collection.find({
            "status": {"$ne": "resolved"},
            "_id": {"$nin": existing_issue_ids}
        }).to_list(1000)
        
        logger.info(f"Fetched {len(issues)} new unresolved issues")

        if not issues:
            return {"status": "no new issues to cluster"}

        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            if "location" not in issue or not issue["location"]:
                logger.warning(f"❌ Issue {issue['_id']} has no location data")
                continue
            coords_str = issue["location"].get("coordinates")
            if not coords_str or not isinstance(coords_str, str):
                logger.warning(f"❌ Issue {issue['_id']} has invalid coordinates format")
                continue
            try:
                lat, lon = map(float, coords_str.split(","))
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    logger.warning(f"❌ Coordinates out of range for issue {issue['_id']}: {coords_str}")
                    continue
            except ValueError as e:
                logger.warning(f"❌ Invalid coordinates format for issue {issue['_id']}: {coords_str}, error: {e}")
                continue
            issue_type = issue.get("type")
            if not issue_type:
                logger.warning(f"❌ Issue {issue['_id']} has no type, skipping")
                continue
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append({
                "issue": issue,
                "coords": [lat, lon]
            })

        results = []
        distance_threshold = 0.5  # 500 meters

        # Process each issue type
        for issue_type, type_issues in issues_by_type.items():
            if len(type_issues) < 2:
                logger.info(f"ℹ️ Skipping clustering for issue type {issue_type} with {len(type_issues)} issues")
                continue

            # Try to add to existing clusters first
            existing_clusters = await clustered_collection.find({"issue_type": issue_type}).to_list(1000)
            unprocessed_issues = type_issues.copy()

            for issue_data in type_issues:
                issue = issue_data["issue"]
                lat, lon = issue_data["coords"]
                issue_id = str(issue["_id"])
                issue_description = issue.get("description", "")
                added_to_existing = False

                for cluster in existing_clusters:
                    cluster_lat, cluster_lon = cluster["location_center"]
                    distance = haversine_distance(lat, lon, cluster_lat, cluster_lon)
                    if distance <= distance_threshold:
                        old_count = cluster["count"]
                        new_count = old_count + 1
                        new_center_lat = (cluster_lat * old_count + lat) / new_count
                        new_center_lon = (cluster_lon * old_count + lon) / new_count

                        all_issue_ids = cluster["issues"] + [issue_id]
                        all_issues = await issues_collection.find(
                            {"_id": {"$in": [ObjectId(i) for i in all_issue_ids]}}
                        ).to_list(None)
                        all_descriptions = [iss.get("description", "") for iss in all_issues]
                        new_summary = await generate_summary(all_descriptions)
                        new_priority = calculate_priority(new_count)

                        await clustered_collection.update_one(
                            {"_id": ObjectId(cluster["_id"])},
                            {
                                "$push": {"issues": issue_id},
                                "$inc": {"count": 1},
                                "$set": {
                                    "location_center": [new_center_lat, new_center_lon],
                                    "general_description": new_summary,
                                    "priority": new_priority,
                                    "updatedAt": datetime.utcnow()
                                }
                            }
                        )
                        logger.info(f"✅ Added issue {issue_id} to existing cluster {cluster['_id']} (issue_type: {issue_type})")
                        added_to_existing = True
                        unprocessed_issues.remove(issue_data)
                        break

            # Apply DBSCAN to remaining unprocessed issues
            if len(unprocessed_issues) < 2:
                logger.info(f"ℹ️ Skipping DBSCAN for issue type {issue_type} with {len(unprocessed_issues)} issues")
                continue

            coords = np.array([item["coords"] for item in unprocessed_issues])
            clustering = DBSCAN(eps=0.005, min_samples=2, metric="haversine").fit(np.radians(coords))
            labels = clustering.labels_

            clusters_dict = {}
            for idx, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(unprocessed_issues[idx])

            for label, cluster_issues in clusters_dict.items():
                if len(cluster_issues) < 2:
                    logger.info(f"ℹ️ Skipping cluster {label} with only {len(cluster_issues)} issues")
                    continue

                center_lat = np.mean([item["coords"][0] for item in cluster_issues])
                center_lon = np.mean([item["coords"][1] for item in cluster_issues])
                descriptions = [item["issue"].get("description", "") for item in cluster_issues]
                issue_ids = [str(item["issue"]["_id"]) for item in cluster_issues]

                new_cluster_id = await clustered_collection.count_documents({}) + 1
                cluster_doc = {
                    "cluster_id": new_cluster_id,
                    "issue_type": issue_type,
                    "issues": issue_ids,
                    "location_center": [float(center_lat), float(center_lon)],
                    "count": len(cluster_issues),
                    "general_description": await generate_summary(descriptions),
                    "priority": calculate_priority(len(cluster_issues)),
                    "createdAt": datetime.utcnow()
                }

                cluster_doc = to_python_types(cluster_doc)
                result = await clustered_collection.insert_one(cluster_doc)
                results.append({"status": "new_cluster", "cluster_doc": serialize_doc(cluster_doc)})
                logger.info(f"✅ Created new cluster with {len(cluster_issues)} issues of type {issue_type}")

        return {"status": "clustering complete", "results": results}

    except Exception as e:
        logger.exception("❌ Error during clustering")
        return {"status": "error", "message": str(e)}

# -------------------
# Get clusters endpoint
# -------------------
@app.get("/clusters")
async def get_clusters():
    clusters = await clustered_collection.find().to_list(100)
    logger.info(f"📊 Retrieved {len(clusters)} clusters from DB")
    return [serialize_doc(c) for c in clusters]