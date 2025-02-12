# How to use MACHO download service

1. Go to the [APIs Console](https://console.developers.google.com/iam-admin/projects) and create a new project.
2. Search for `Google Drive API`, select it, and click `Enable`.
3. In the left menu, select `Credentials`, click `Create Credentials`, and choose `OAuth client ID`.

Download your credentials file and place it in the root directory (e.g., `./`).

Then, execute the script `run.bash`.

You will be prompted to authenticate by following a link that opens in your browser. Follow the instructions and accept the terms. Finally, the browser will attempt to redirect you to a localhost URL. The URL should look like this:
```
http://localhost/?code=4/0AVG7fiQgnPRs-3Wy5tKJSCAh_ozRiiF8naDCcs0d-ei9YQs_BFuJ8aTH_T2kxGmDL_wdmeQ&scope=https://www.googleapis.com/auth/drive
```

Copy and paste the code that appears after the `code=` tag. In the example above, the code would be:
```
4/0AVG7fiQgnPRs-3Wy5tKJSCAh_ozRiiF8naDCcs0d-ei9YQs_BFuJ8aTH_T2kxGmDL_wdmeQ
```
