use crate::cloud::credentials::CloudCredentials;
use crate::error::{LlmZipError, Result};
use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
use std::sync::Arc;

/// Parsed S3 destination descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Uri {
    pub bucket: String,
    pub key: String,
}

impl S3Uri {
    /// Parses an `s3://bucket/key` URI string.
    pub fn parse(uri: &str) -> Result<Self> {
        let trimmed = uri.trim();
        if !trimmed.starts_with("s3://") {
            return Err(LlmZipError::SecurityPathViolation {
                offending_path: uri.to_string(),
                reason: "URI must start with 's3://'".to_string(),
            });
        }

        let without_scheme = &trimmed[5..];
        let mut parts = without_scheme.splitn(2, '/');
        let bucket = parts.next().unwrap_or("").trim().to_string();
        let key = parts.next().unwrap_or("").trim().to_string();

        if bucket.is_empty() || key.is_empty() {
            return Err(LlmZipError::SecurityPathViolation {
                offending_path: uri.to_string(),
                reason: "S3 URI must specify both bucket and object key (e.g., s3://bucket/path/model.gguf)".to_string(),
            });
        }

        Ok(Self { bucket, key })
    }
}

/// Builds a configured `ObjectStore` client targeting the parsed S3/R2 destination.
pub fn build_s3_store(target: &S3Uri, creds: Option<&CloudCredentials>) -> Result<Arc<dyn ObjectStore>> {
    let mut builder = AmazonS3Builder::new().with_bucket_name(&target.bucket);

    if let Some(c) = creds {
        builder = builder
            .with_access_key_id(&c.access_key_id)
            .with_secret_access_key(&c.secret_access_key)
            .with_region(&c.region);

        if let Some(token) = &c.session_token {
            builder = builder.with_token(token);
        }

        if let Some(endpoint) = &c.endpoint_url {
            builder = builder.with_endpoint(endpoint);
            if endpoint.starts_with("http://") {
                builder = builder.with_allow_http(true);
            }
        }
    } else {
        // Fallback to automatic environment / instance metadata resolution
        builder = builder.with_region("us-east-1");
    }

    let s3 = builder
        .build()
        .map_err(|e| LlmZipError::S3Error {
            bucket: target.bucket.clone(),
            key: target.key.clone(),
            reason: format!("Failed to configure S3 client: {}", e),
        })?;

    Ok(Arc::new(s3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_uri_parse() {
        let uri = S3Uri::parse("s3://my-bucket/models/llama-3.gguf").unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.key, "models/llama-3.gguf");

        assert!(S3Uri::parse("http://my-bucket/key").is_err());
        assert!(S3Uri::parse("s3://my-bucket").is_err());
        assert!(S3Uri::parse("s3://my-bucket/").is_err());
    }
}
