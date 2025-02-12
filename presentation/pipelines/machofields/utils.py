from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def connect_to_drive():
    gauth = GoogleAuth()

    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        # Authenticate if they're not there
        print('[INFO] Need to authenticate')
        # This is what solved the issues:
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline'})
        gauth.flow.params.update({'approval_prompt': 'force'})
        # gauth.LocalWebserverAuth(host_name='0.0.0.0', port_numbers=[8080], headless=True)
        auth_url = gauth.GetAuthUrl()
        print(f"Please visit the following URL and authenticate: {auth_url}")

        # Wait for the user to complete the authentication process
        code = input("Enter the authorization code: ")
        gauth.Auth(code)

    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()

    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")  
    drive = GoogleDrive(gauth)

    return drive