resource "aws_budgets_budget" "this" {
  name         = var.name
  budget_type  = "COST"
  limit_amount = tostring(var.limit_amount)
  limit_unit   = var.limit_unit
  time_unit    = var.time_unit

  dynamic "notification" {
    for_each = local.notification_thresholds
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "ABSOLUTE_VALUE"
      notification_type          = var.notification_type
      subscriber_email_addresses = var.subscriber_email_addresses
    }
  }
}

locals {
  notification_thresholds = sort(distinct(var.notification_thresholds))
}
