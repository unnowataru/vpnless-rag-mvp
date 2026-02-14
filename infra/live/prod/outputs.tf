output "budget_name" {
  description = "Budget name"
  value       = module.budget.name
}

output "bedrock_invoker_user_name" {
  description = "Bedrock invoker IAM user name"
  value       = module.rag_bedrock_invoker.user_name
}

output "bedrock_invoker_user_arn" {
  description = "Bedrock invoker IAM user ARN"
  value       = module.rag_bedrock_invoker.user_arn
}

output "bedrock_invoker_policy_arn" {
  description = "Bedrock invoker IAM policy ARN"
  value       = module.rag_bedrock_invoker.policy_arn
}
