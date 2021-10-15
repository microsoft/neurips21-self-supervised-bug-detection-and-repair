
resource "kubernetes_namespace" "gpu_resources" {
    metadata {
        annotations = {name = "gpu-resources"}
        name = "gpu-resources"
    }
}

resource "kubernetes_namespace" "monitoring" {
    metadata {
        annotations = {name = "monitoring"}
        name = "monitoring"
    }
}

# Helm
provider "helm" {
    alias = "aks"
    kubernetes {
        host                   = azurerm_kubernetes_cluster.buglab.kube_config.0.host
        username               = azurerm_kubernetes_cluster.buglab.kube_config.0.username
        password               = azurerm_kubernetes_cluster.buglab.kube_config.0.password

        client_key             = base64decode(azurerm_kubernetes_cluster.buglab.kube_config.0.client_key)
        client_certificate     = base64decode(azurerm_kubernetes_cluster.buglab.kube_config.0.client_certificate)
        cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.buglab.kube_config.0.cluster_ca_certificate)
        load_config_file       = false
    }
}

resource "helm_release" "blob" {
    name       = "blob-csi-driver"
    chart      = "blob-csi-driver"
    version    = "0.10.0"
    repository = "https://raw.githubusercontent.com/kubernetes-sigs/blob-csi-driver/master/charts"
    namespace  = "kube-system"
}

resource "helm_release" "gpu-management" {
    name       = "nvidia-device-plugin"
    chart      = "nvidia-device-plugin"
    version    = "0.7.1"
    repository = "https://nvidia.github.io/k8s-device-plugin"
    namespace  = "gpu-resources"
    values = [
        file("nvidia_values.yaml")
    ]
}

resource "helm_release" "prometheus" {
    chart      = "kube-prometheus-stack"
    name       = "kube-prometheus-stack"
    repository = "https://prometheus-community.github.io/helm-charts"
    version    = "12.2.3"
    namespace  = "monitoring"
    values = [
        file("prometheus_values.yaml")
    ]
}
