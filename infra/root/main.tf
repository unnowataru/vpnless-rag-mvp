resource "aws_budgets_budget" "monthly" {
  name         = "vpnless-rag-mvp-monthly"
  budget_type  = "COST"
  limit_amount = "90"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 45
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["unnow@networld.co.jp"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 70
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["unnow@networld.co.jp"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 85
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["unnow@networld.co.jp"]
  }
}

# --- Bedrock invoker (app/on-prem) ---
resource "aws_iam_user" "bedrock_invoker" {
  name = "rag-bedrock-invoker"
}

data "aws_iam_policy_document" "bedrock_invoker" {
  statement {
    sid    = "InvokeBedrockModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]
    resources = ["arn:aws:bedrock:ap-northeast-1::foundation-model/google.gemma-3-4b-it"]
  }

  # Optional: allow listing models for troubleshooting
  statement {
    sid       = "ListModels"
    effect    = "Allow"
    actions   = ["bedrock:ListFoundationModels"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "bedrock_invoker" {
  name   = "rag-bedrock-invoker-policy"
  policy = data.aws_iam_policy_document.bedrock_invoker.json
}

resource "aws_iam_user_policy_attachment" "bedrock_invoker" {
  user       = aws_iam_user.bedrock_invoker.name
  policy_arn = aws_iam_policy.bedrock_invoker.arn
}
