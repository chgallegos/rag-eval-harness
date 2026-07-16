# API Rate Limits

The standard API tier allows 100 requests per minute per key. The enterprise
tier allows 1000 requests per minute. When a limit is exceeded the API
returns HTTP 429 with a Retry-After header.

Rate limits reset on a rolling 60 second window. Batch endpoints count each
item in the batch as one request.
