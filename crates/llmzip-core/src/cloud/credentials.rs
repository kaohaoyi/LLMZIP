use std::fmt;

/// Secure in-memory representation of cloud object storage credentials.
/// Implements strict redaction on Debug and Display to prevent credential leakage into logs or WAL.
#[derive(Clone)]
pub struct CloudCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub endpoint_url: Option<String>,
    pub region: String,
}

impl CloudCredentials {
    /// Attempts to load credentials from standard environment variables:
    /// - `AWS_ACCESS_KEY_ID`
    /// - `AWS_SECRET_ACCESS_KEY`
    /// - `AWS_SESSION_TOKEN`
    /// - `AWS_ENDPOINT_URL` (for R2 / MinIO)
    /// - `AWS_REGION` / `AWS_DEFAULT_REGION`
    pub fn from_env() -> Option<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        let endpoint_url = std::env::var("AWS_ENDPOINT_URL")
            .ok()
            .or_else(|| std::env::var("S3_ENDPOINT_URL").ok());
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        Some(Self {
            access_key_id,
            secret_access_key,
            session_token,
            endpoint_url,
            region,
        })
    }

    /// Sanitizes an input message by redacting the secret key and session token if present.
    pub fn sanitize(&self, text: &str) -> String {
        let mut clean = text.to_string();
        if !self.secret_access_key.is_empty() {
            clean = clean.replace(&self.secret_access_key, "***REDACTED_SECRET***");
        }
        if let Some(token) = &self.session_token {
            if !token.is_empty() {
                clean = clean.replace(token, "***REDACTED_TOKEN***");
            }
        }
        clean
    }
}

impl fmt::Debug for CloudCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudCredentials")
            .field("access_key_id", &self.access_key_id)
            .field("secret_access_key", &"***REDACTED***")
            .field("session_token", &self.session_token.as_ref().map(|_| "***REDACTED***"))
            .field("endpoint_url", &self.endpoint_url)
            .field("region", &self.region)
            .finish()
    }
}

impl fmt::Display for CloudCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CloudCredentials(id={}, region={}, endpoint={:?})",
            self.access_key_id, self.region, self.endpoint_url
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_redaction() {
        let creds = CloudCredentials {
            access_key_id: "AKIA_TEST_KEY".to_string(),
            secret_access_key: "SuperSecretKey123".to_string(),
            session_token: Some("TokenXYZ".to_string()),
            endpoint_url: Some("https://r2.cloudflarestorage.com".to_string()),
            region: "auto".to_string(),
        };

        let debug_str = format!("{:?}", creds);
        assert!(!debug_str.contains("SuperSecretKey123"));
        assert!(!debug_str.contains("TokenXYZ"));
        assert!(debug_str.contains("***REDACTED***"));

        let log_msg = "Error connecting with SuperSecretKey123 and TokenXYZ";
        let clean = creds.sanitize(log_msg);
        assert!(!clean.contains("SuperSecretKey123"));
        assert!(!clean.contains("TokenXYZ"));
        assert_eq!(clean, "Error connecting with ***REDACTED_SECRET*** and ***REDACTED_TOKEN***");
    }
}
