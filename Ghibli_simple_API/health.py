import requests
import time

# Service health check endpoint configuration
API_HEALTH_URL = "http://localhost:7860/health"  # Default local address
TIMEOUT = 10  # Request timeout in seconds

def check_health():
    """
    Call service health check interface
    Returns:
        dict: Dictionary containing service status and details
    """
    try:
        # Send GET request
        response = requests.get(API_HEALTH_URL, timeout=TIMEOUT)
        response.raise_for_status()  # Automatically trigger HTTP error exception
        
        # Parse JSON response
        health_data = response.json()
        return {
            "online": True,
            "status": health_data.get("status", "unknown"),
            "model_loaded": health_data.get("model_loaded", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        return {
            "online": False,
            "error_type": "network_error",
            "error_message": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except json.JSONDecodeError:
        # Handle invalid JSON response
        return {
            "online": False,
            "error_type": "invalid_response",
            "error_message": "Received non-JSON response",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def print_health_status(status_data):
    """
    Print health status in a visual format
    """
    if status_data["online"]:
        print(f"[✅] Service online (Check time: {status_data['timestamp']})")
        print(f"    - Status: {status_data['status'].upper()}")
        print(f"    - Model loaded: {'Success' if status_data['model_loaded'] else 'Failed'}")
    else:
        print(f"[❌] Service unreachable (Check time: {status_data['timestamp']})")
        print(f"    Error type: {status_data.get('error_type', 'unknown')}")
        print(f"    Error message: {status_data.get('error_message', 'No details')}")

if __name__ == "__main__":
    health_status = check_health()
    print_health_status(health_status)