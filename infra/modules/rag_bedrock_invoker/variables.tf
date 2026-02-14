variable "user_name" {
  description = "IAM user name for Bedrock invocation"
  type        = string
}

variable "policy_name" {
  description = "IAM policy name attached to the invoker user"
  type        = string
}

variable "region" {
  description = "AWS region for model ARNs"
  type        = string
}

variable "allowed_model_ids" {
  description = "Bedrock model IDs allowed for InvokeModel/InvokeModelWithResponseStream"
  type        = list(string)

  validation {
    condition     = length(var.allowed_model_ids) > 0
    error_message = "allowed_model_ids must contain at least one model id."
  }
}

variable "rerank_model_id" {
  description = "Bedrock rerank model ID"
  type        = string
}

variable "allow_list_models" {
  description = "Whether to allow bedrock:ListFoundationModels"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to assign to IAM resources"
  type        = map(string)
  default     = {}
}
