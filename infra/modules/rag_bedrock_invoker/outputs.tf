output "user_name" {
  value       = aws_iam_user.this.name
  description = "IAM user name"
}

output "user_arn" {
  value       = aws_iam_user.this.arn
  description = "IAM user ARN"
}

output "policy_arn" {
  value       = aws_iam_policy.this.arn
  description = "IAM policy ARN"
}
