
## LockProcess.py
takes as input:
* A directory (either on S3 or local)
* A regular expression
* a script to run on each file

Looks for a file without a lock, creates a lock, copies the file to
local directory and calls the script to process it.

### delS3.py:
Removes all of the files particular S3 directory and pattern.

LocalLock:
