locals {
  service = "gemini-agent-hub"
  apis = [
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "logging.googleapis.com",
    "bigquery.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "repo" {
  repository_id = "agent-hub"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ---------- least-privilege runtime identity ----------
resource "google_service_account" "runtime" {
  account_id   = "agent-hub-runtime"
  display_name = "Gemini Agent Hub runtime"
}

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset([
    "roles/aiplatform.user",       # call Gemini + embeddings on Vertex AI
    "roles/logging.logWriter",     # structured audit logs
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# ---------- Cloud Run service ----------
resource "google_cloud_run_v2_service" "hub" {
  name                = local.service
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }
    max_instance_request_concurrency = 40
    timeout                          = "120s"

    containers {
      image = var.image
      resources {
        limits            = { cpu = "1", memory = "1Gi" }
        cpu_idle          = true
        startup_cpu_boost = true
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "AGENT_MODEL"
        value = var.model
      }
      startup_probe {
        http_get { path = "/healthz" }
        initial_delay_seconds = 2
        period_seconds        = 5
        failure_threshold     = 12
      }
      liveness_probe {
        http_get { path = "/healthz" }
        period_seconds = 30
      }
    }
  }
  depends_on = [google_project_service.apis]
}

# Private by default: only listed members (or IAP) may invoke.
resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = toset(var.invoker_members)
  name     = google_cloud_run_v2_service.hub.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}

# ---------- audit trail: guardrail + tool events -> BigQuery ----------
resource "google_bigquery_dataset" "audit" {
  dataset_id = "agent_hub_audit"
  location   = var.region
  depends_on = [google_project_service.apis]
}

resource "google_logging_project_sink" "audit" {
  name                   = "agent-hub-audit"
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit.dataset_id}"
  filter                 = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.service}\" AND jsonPayload.component=\"gemini-agent-hub\""
  unique_writer_identity = true
  bigquery_options { use_partitioned_tables = true }
}

resource "google_bigquery_dataset_iam_member" "sink_writer" {
  dataset_id = google_bigquery_dataset.audit.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.audit.writer_identity
}
