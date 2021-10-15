# Please note
Changes to the definitions in these files will **not** update the dashboards that you see in the
Grafana instance for your experiment if you are running it using the `helm install` method detailed
in the main README.

To update those dashboards, you need to sync your changes with the files stored in the `grafana`
blob container in the storage account defined in Terraform.
