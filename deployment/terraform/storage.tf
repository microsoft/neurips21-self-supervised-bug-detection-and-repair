resource "azurerm_storage_account" "buglab" {
    name                     = "buglab"
    resource_group_name      = azurerm_resource_group.buglab.name
    location                 = azurerm_resource_group.buglab.location
    account_tier             = "Standard"
    account_replication_type = "LRS"
    allow_blob_public_access = "true"
    min_tls_version = "TLS1_2"

    tags = {
        ms-resource-usage = "azure-cloud-shell"
    }

    network_rules {
        default_action = "Allow"
    }
}

resource "azurerm_storage_container" "buglab" {
    name                  = "data"
    storage_account_name  = azurerm_storage_account.buglab.name
    container_access_type = "private"
}

resource "azurerm_storage_container" "grafana" {
    name                  = "grafana"
    storage_account_name  = azurerm_storage_account.buglab.name
    container_access_type = "private"
}

resource "azurerm_storage_container" "tfstate" {
    name                  = "tfstate"
    storage_account_name  = azurerm_storage_account.buglab.name
    container_access_type = "private"
}
