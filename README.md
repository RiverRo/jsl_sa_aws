# jsl_sa_aws

## Configuration

```python
from datetime import datetime, timezone
from jsa_utils import create_directories

PORT=1234
HOST=''
utc_date = datetime.now(timezone.utc).date().isoformat()  # ignored by sanic b/c lowercase key
LOG_FILENAME=f'./logs/jslsa_{utc_date}.log'
create_directories(LOG_FILENAME)
```
