source docker-compose.env
colonies fs sync -l models -d ./models --yes
colonies fs sync -l my_cloud_filtering -d ./my_cloud_filtering --yes
colonies fs sync -l outputs -d ./outputs --yes 