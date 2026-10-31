terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

# 1. S3 Model Registry for Quantized YOLO26 + Detectron2 ONNX Artifacts & Drift Logs
resource "aws_s3_bucket" "model_registry" {
  bucket_prefix = "aegisvision-model-registry-${var.environment}-"
}

resource "aws_s3_bucket_versioning" "model_registry_versioning" {
  bucket = aws_s3_bucket.model_registry.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 2. Amazon SageMaker Serverless Inference Endpoint (Scales to Zero when Idle)
resource "aws_iam_role" "sagemaker_exec_role" {
  name = "aegisvision-sagemaker-serverless-role-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_sagemaker_model" "aegisvision_onnx_int8" {
  name               = "aegisvision-yolo26-detectron2-onnx"
  execution_role_arn = aws_iam_role.sagemaker_exec_role.arn

  primary_container {
    image          = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/pytorch-inference:2.4.0-cpu-py311"
    model_data_url = "s3://${aws_s3_bucket.model_registry.bucket}/models/aegisvision-int8-model.tar.gz"
    environment = {
      OMP_NUM_THREADS       = "4"
      ORT_TENSORRT_FP16_ENABLE = "1"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "serverless_cv_config" {
  name = "aegisvision-serverless-cv-config-${var.environment}"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.aegisvision_onnx_int8.name

    serverless_config {
      max_concurrency   = 10
      memory_size_in_mb = 4096
    }
  }
}

resource "aws_sagemaker_endpoint" "serverless_cv_endpoint" {
  name                 = "aegisvision-cv-serverless-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.serverless_cv_config.name
}

# 3. AWS ECS Fargate Cluster for FastAPI + LangGraph Hybrid RAG CAPA Orchestrator
resource "aws_ecs_cluster" "aegisvision_cluster" {
  name = "aegisvision-ai-cluster-${var.environment}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

output "sagemaker_serverless_endpoint_name" {
  value = aws_sagemaker_endpoint.serverless_cv_endpoint.name
}

output "model_registry_bucket" {
  value = aws_s3_bucket.model_registry.bucket
}
