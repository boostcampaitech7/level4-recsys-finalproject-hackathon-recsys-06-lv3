source .env
scp -r $data_path server$server_no:/data/ephemeral/home/$git_project_name
ssh server$server_no << ENDSSH
cd /data/ephemeral/home/$git_project_name/data
ls -lta
ENDSSH
./3.start-app.sh