variable "project_id" {
  type        = string
  description = "GCP project to deploy into."
}

variable "region" {
  type        = string
  default     = "asia-south1"
  description = "Region for Cloud Run, Artifact Registry and Vertex AI."
}

variable "image" {
  type        = string
  description = "Container image URI, e.g. asia-south1-docker.pkg.dev/PROJECT/agent-hub/gemini-agent-hub:TAG"
}

variable "invoker_members" {
  type        = list(string)
  default     = []
  description = "IAM members allowed to call the service, e.g. [\"group:eng@example.com\"]."
}

variable "model" {
  type    = string
  default = "gemini-2.5-flash"
}

variable "max_instances" {
  type    = number
  default = 5
}
