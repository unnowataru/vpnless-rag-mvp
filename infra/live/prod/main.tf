module "budget" {
  source = "../../modules/budget"

  name                       = var.budget_name
  limit_amount               = var.budget_limit_amount
  notification_thresholds    = var.budget_notification_thresholds
  subscriber_email_addresses = var.budget_subscriber_emails
}

module "rag_bedrock_invoker" {
  source = "../../modules/rag_bedrock_invoker"

  user_name         = var.bedrock_invoker_user_name
  policy_name       = var.bedrock_invoker_policy_name
  region            = var.aws_region
  allowed_model_ids = var.bedrock_allowed_model_ids
  rerank_model_id   = var.bedrock_rerank_model_id
  allow_list_models = var.bedrock_allow_list_models
  tags              = var.tags
}
