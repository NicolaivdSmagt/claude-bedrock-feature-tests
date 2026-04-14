# ABOUTME: Terraform configuration for the feature scout agent's SNS notification topic.
# ABOUTME: Creates an SNS topic and email subscription for daily/weekly agent reports.

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

variable "aws_region" {
  description = "AWS region for the SNS topic"
  type        = string
}

variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
}

variable "notification_email" {
  description = "Email address to receive agent reports (must be confirmed after apply)"
  type        = string
}

variable "topic_name" {
  description = "Name for the SNS topic"
  type        = string
  default     = "bedrock-feature-scout"
}

resource "aws_sns_topic" "feature_scout" {
  name = var.topic_name
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.feature_scout.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

output "sns_topic_arn" {
  description = "SNS topic ARN — add this to config.yaml as sns_topic_arn"
  value       = aws_sns_topic.feature_scout.arn
}
