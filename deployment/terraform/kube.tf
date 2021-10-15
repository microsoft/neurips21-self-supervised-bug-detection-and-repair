resource "azurerm_log_analytics_workspace" "buglab" {
    # The WorkSpace name has to be unique across the whole of azure, not just the current subscription/tenant.
    name                = var.log_analytics_workspace_name
    location            = var.log_analytics_workspace_location
    resource_group_name = azurerm_resource_group.buglab.name
    sku                 = var.log_analytics_workspace_sku
}

resource "azurerm_log_analytics_solution" "buglab" {
    solution_name         = "ContainerInsights"
    location              = azurerm_log_analytics_workspace.buglab.location
    resource_group_name   = azurerm_resource_group.buglab.name
    workspace_resource_id = azurerm_log_analytics_workspace.buglab.id
    workspace_name        = azurerm_log_analytics_workspace.buglab.name

    plan {
        publisher = "Microsoft"
        product   = "OMSGallery/ContainerInsights"
    }
}

resource "azurerm_kubernetes_cluster" "buglab" {
    name                = "${var.prefix}-k8s"
    location            = azurerm_resource_group.buglab.location
    resource_group_name = azurerm_resource_group.buglab.name
    dns_prefix          = "${var.prefix}-k8s"

    service_principal {
        client_id     = var.client_id
        client_secret = var.client_secret
    }

    default_node_pool {
        name       = "default"
        node_count = var.agent_count
        vm_size    = "Standard_D2s_v3"
        vnet_subnet_id = azurerm_subnet.internal.id
    }

    addon_profile {
        oms_agent {
            enabled                    = true
            log_analytics_workspace_id = azurerm_log_analytics_workspace.buglab.id
        }
        kube_dashboard {
            enabled = true
        }
    }

    network_profile {
        load_balancer_sku = "Standard"
        network_plugin = "kubenet"
    }

    tags = {
        Environment = "Development"
    }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.buglab.id
  vm_size               = "Standard_NC6s_v2"
  max_count             = 50
  min_count             = 0
  enable_auto_scaling   = true
  vnet_subnet_id = azurerm_subnet.internal.id
  node_taints = ["compute=gpu:NoSchedule"]
  node_labels = {
      compute = "gpu"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "cpu" {
  name                  = "cpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.buglab.id
  vm_size               = "Standard_D64s_v3"
  max_count             = 100
  min_count             = 0
  enable_auto_scaling   = true

  vnet_subnet_id        = azurerm_subnet.internal.id
  node_labels = {
    compute = "cpu"
  }
}

resource "kubernetes_secret" "azure-secret" {
  metadata {
    name = "azure-secret"
  }
  type = "Opaque"

  data = {
      azurestorageaccountname = azurerm_storage_account.buglab.name
      azurestorageaccountkey = var.storage_account_key
  }

}
