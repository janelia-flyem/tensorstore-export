"""
GCS bucket inspection and configuration utilities.

Used by deploy.py to validate and configure destination buckets for
export workloads.  Centralizes all bucket metadata operations so that
deploy.py and export.py stay focused on their own responsibilities.
"""

from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions


def get_bucket_info(bucket_name: str, project: str = "") -> dict | None:
    """Fetch bucket metadata from GCS.

    Returns a dict with location, location_type, storage_class,
    soft_delete_retention_seconds, and versioning_enabled.
    Returns None if the bucket does not exist.
    Returns a dict with an 'error' key on permission errors.
    """
    client = storage.Client(project=project or None)
    bucket = client.bucket(bucket_name)
    try:
        bucket.reload()
    except gcs_exceptions.NotFound:
        return None
    except gcs_exceptions.Forbidden as e:
        return {"error": str(e)}

    soft_delete_seconds = 0
    if bucket.soft_delete_policy:
        soft_delete_seconds = (
            bucket.soft_delete_policy.retention_duration_seconds or 0
        )

    return {
        "location": bucket.location,
        "location_type": bucket.location_type,
        "storage_class": bucket.storage_class,
        "soft_delete_retention_seconds": soft_delete_seconds,
        "versioning_enabled": bool(bucket.versioning_enabled),
    }


def create_bucket(bucket_name: str, location: str, project: str = "") -> bool:
    """Create a single-region GCS bucket with export-safe defaults.

    Creates the bucket with:
    - Single-region in the specified location
    - Standard storage class
    - Uniform bucket-level access
    - Hierarchical namespace enabled
    - Soft delete disabled

    Returns True on success, False on failure (prints error).
    """
    client = storage.Client(project=project or None)
    bucket = client.bucket(bucket_name)
    bucket.location = location
    bucket.storage_class = "STANDARD"
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.hierarchical_namespace_enabled = True

    try:
        client.create_bucket(bucket)
    except gcs_exceptions.Conflict:
        print(f"  Error: bucket '{bucket_name}' already exists (globally unique name conflict).")
        return False
    except gcs_exceptions.Forbidden as e:
        print(f"  Error: permission denied creating bucket '{bucket_name}': {e}")
        return False

    # Disable soft delete (must be done after creation)
    try:
        bucket.soft_delete_policy.retention_duration_seconds = 0
        bucket.patch()
    except Exception as e:
        print(f"  Warning: bucket created but could not disable soft delete: {e}")
        print(f"  Run manually: gcloud storage buckets update gs://{bucket_name} --no-soft-delete")

    return True


def disable_soft_delete(bucket_name: str, project: str = "") -> int:
    """Disable soft delete on a bucket.

    Returns the previous retention_duration_seconds (0 if already disabled).
    Raises on permission errors with a helpful message.
    """
    client = storage.Client(project=project or None)
    bucket = client.bucket(bucket_name)
    bucket.reload()

    old_retention = 0
    if bucket.soft_delete_policy:
        old_retention = bucket.soft_delete_policy.retention_duration_seconds or 0

    if old_retention > 0:
        try:
            bucket.soft_delete_policy.retention_duration_seconds = 0
            bucket.patch()
        except gcs_exceptions.Forbidden:
            raise PermissionError(
                f"Permission denied disabling soft delete on '{bucket_name}'.\n"
                f"  Run manually: gcloud storage buckets update gs://{bucket_name} --no-soft-delete"
            )

    return old_retention


def check_write_permission(bucket_name: str, prefix: str = "",
                           project: str = "") -> bool:
    """Test that the default service account can write to the bucket.

    Writes a small test object and deletes it.  Returns True if
    successful, False on permission error (prints the error).
    """
    client = storage.Client(project=project or None)
    test_path = f"{prefix}/.deploy-permission-check".lstrip("/")
    blob = client.bucket(bucket_name).blob(test_path)
    try:
        blob.upload_from_string(b"permission check", content_type="text/plain")
        blob.delete()
        return True
    except gcs_exceptions.Forbidden:
        print(f"  Error: write permission denied on 'gs://{bucket_name}/{test_path}'")
        print("  The Cloud Run service account needs storage.objects.create access.")
        print("  Grant it with:")
        print(f"    gcloud storage buckets add-iam-policy-binding gs://{bucket_name} \\")
        print("      --member='serviceAccount:<SERVICE_ACCOUNT>' \\")
        print("      --role='roles/storage.objectAdmin'")
        return False


def check_read_permission(bucket_name: str, prefix: str = "",
                          project: str = "") -> bool:
    """Test that the default service account can read from the bucket.

    Lists objects under the prefix.  Returns True if successful,
    False on permission error (prints the error).
    """
    client = storage.Client(project=project or None)
    bucket = client.bucket(bucket_name)
    try:
        # Just try to list one object — this tests storage.objects.list
        next(iter(bucket.list_blobs(prefix=prefix, max_results=1)), None)
        return True
    except gcs_exceptions.Forbidden:
        print(f"  Error: read permission denied on 'gs://{bucket_name}/{prefix}'")
        print("  The Cloud Run service account needs storage.objects.get access.")
        print("  Grant it with:")
        print(f"    gcloud storage buckets add-iam-policy-binding gs://{bucket_name} \\")
        print("      --member='serviceAccount:<SERVICE_ACCOUNT>' \\")
        print("      --role='roles/storage.objectViewer'")
        return False


def validate_bucket_region(bucket_info: dict, compute_region: str,
                           bucket_name: str) -> list[str]:
    """Check bucket region against compute region.

    Returns a list of warning strings (empty if everything is fine).
    """
    warnings = []
    location = bucket_info["location"].upper()
    location_type = bucket_info.get("location_type", "").lower()
    compute_upper = compute_region.upper()

    if location_type in ("multi-region", "dual-region"):
        warnings.append(
            f"Bucket '{bucket_name}' is {location_type} ({location}). "
            f"Cloud Run runs in {compute_region}. Multi-region buckets incur "
            f"cross-region replication charges at $0.02/GiB on every write. "
            f"Use a single-region bucket in {compute_region} instead."
        )
    elif location != compute_upper:
        warnings.append(
            f"Bucket '{bucket_name}' is in {location} but Cloud Run runs in "
            f"{compute_region}. Cross-region egress charges will apply."
        )

    return warnings
