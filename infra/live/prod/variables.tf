variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "aws_profile" {
  description = "AWS profile for Terraform provider (null uses env/default chain)"
  type        = string
  default     = null
  nullable    = true
}

variable "budget_name" {
  description = "Budget resource name"
  type        = string
  default     = "vpnless-rag-mvp-monthly"
}

variable "budget_limit_amount" {
  description = "Monthly budget amount"
  type        = number
  default     = 90
}

variable "budget_notification_thresholds" {
  description = "Absolute budget thresholds for notification"
  type        = list(number)
  default     = [45, 70, 85]
}

variable "budget_subscriber_emails" {
  description = "Notification destination emails"
  type        = list(string)
}

variable "bedrock_invoker_user_name" {
  description = "IAM user name for Bedrock invocation"
  type        = string
  default     = "rag-bedrock-invoker"
}

variable "bedrock_invoker_policy_name" {
  description = "IAM policy name for Bedrock invoker"
  type        = string
  default     = "rag-bedrock-invoker-policy"
}

variable "bedrock_allowed_model_ids" {
  description = "Allowed Bedrock model IDs for invocation"
  type        = list(string)
  default = [
    "google.gemma-3-4b-it",
    "google.gemma-3-27b-it",
    "amazon.titan-embed-text-v2:0"
  ]
}

variable "bedrock_rerank_model_id" {
  description = "Bedrock rerank model ID"
  type        = string
  default     = "amazon.rerank-v1:0"
}

variable "bedrock_allow_list_models" {
  description = "Allow bedrock:ListFoundationModels"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}
