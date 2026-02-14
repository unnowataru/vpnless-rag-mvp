variable "name" {
  description = "Budget name"
  type        = string
}

variable "limit_amount" {
  description = "Monthly budget limit amount"
  type        = number
}

variable "limit_unit" {
  description = "Budget currency"
  type        = string
  default     = "USD"
}

variable "time_unit" {
  description = "Budget period"
  type        = string
  default     = "MONTHLY"
}

variable "notification_type" {
  description = "Budget notification type"
  type        = string
  default     = "ACTUAL"
}

variable "notification_thresholds" {
  description = "Absolute thresholds for notifications"
  type        = list(number)

  validation {
    condition     = length(var.notification_thresholds) > 0
    error_message = "notification_thresholds must contain at least one value."
  }
}

variable "subscriber_email_addresses" {
  description = "Email recipients for budget notifications"
  type        = list(string)

  validation {
    condition     = length(var.subscriber_email_addresses) > 0
    error_message = "subscriber_email_addresses must contain at least one email."
  }
}
