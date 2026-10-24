output "service_url" {
  value = google_cloud_run_v2_service.hub.uri
}

output "runtime_service_account" {
  value = google_service_account.runtime.email
}

output "audit_dataset" {
  value = google_bigquery_dataset.audit.dataset_id
}
