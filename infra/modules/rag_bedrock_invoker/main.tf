locals {
  allowed_model_arns = [for model_id in var.allowed_model_ids : "arn:aws:bedrock:${var.region}::foundation-model/${model_id}"]
  rerank_model_arn   = "arn:aws:bedrock:${var.region}::foundation-model/${var.rerank_model_id}"
}

resource "aws_iam_user" "this" {
  name = var.user_name
  tags = var.tags

  lifecycle {
    ignore_changes = [tags]
  }
}

data "aws_iam_policy_document" "this" {
  statement {
    sid    = "InvokeBedrockModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]
    resources = local.allowed_model_arns
  }

  statement {
    sid    = "InvokeRerankerModel"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel"
    ]
    resources = [local.rerank_model_arn]
  }

  statement {
    sid       = "RerankDocuments"
    effect    = "Allow"
    actions   = ["bedrock:Rerank"]
    resources = ["*"]
  }

  dynamic "statement" {
    for_each = var.allow_list_models ? [1] : []
    content {
      sid       = "ListModels"
      effect    = "Allow"
      actions   = ["bedrock:ListFoundationModels"]
      resources = ["*"]
    }
  }
}

resource "aws_iam_policy" "this" {
  name   = var.policy_name
  policy = data.aws_iam_policy_document.this.json
  tags   = var.tags
}

resource "aws_iam_user_policy_attachment" "this" {
  user       = aws_iam_user.this.name
  policy_arn = aws_iam_policy.this.arn
}
