# Configure the Azure provider
terraform {
    required_providers {
        azurerm = {
            source = "hashicorp/azurerm"
            version = ">= 2.26"
        }
    }
    backend "azurerm" {
        resource_group_name = "buglab"
        storage_account_name = "buglab"
        container_name = "tfstate"
        key = "terraform.tfstate"
    }
}

provider "azurerm" {
    features {}
    subscription_id = var.subscription_id
    tenant_id = var.tennant_id
    skip_provider_registration = true
    client_id = var.client_id
    client_secret = var.client_secret
}

resource "azurerm_resource_group" "buglab" {
    name     = var.resource_group_name
    location = var.location
}
