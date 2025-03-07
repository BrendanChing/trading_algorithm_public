// Register the service worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js') // Point to root-level file
        .then(function (registration) {
            console.log('Service Worker registered with scope:', registration.scope);
        })
        .catch(function (error) {
            console.error('Service Worker registration failed:', error);
        });
}

function subscribeUser() {
    console.log('subscribeUser function is executing');

    if ('serviceWorker' in navigator && 'PushManager' in window) {
        console.log('Service Worker and PushManager are supported');

        navigator.serviceWorker.ready
            .then(function (registration) {
                console.log('Service Worker is ready:', registration);

                registration.pushManager.getSubscription()
                    .then(function (existingSubscription) {
                        if (existingSubscription) {
                            console.log('User already subscribed:', existingSubscription.toJSON());
                            return; // User is already subscribed
                        }

                        console.log('No existing subscription found, proceeding to subscribe user');
                        
                        // Subscribe the user
                        const appServerKey = urlBase64ToUint8Array('BBS10xMUSLRw3g9PhIGFner4uQbPYfcSTQ8vF3RMSa6JO6DDJ4fwgYr1k6AtqAkyYPMxB7F9CikPHINnHaPix8c');

                        registration.pushManager.subscribe({
                            userVisibleOnly: true,
                            applicationServerKey: appServerKey
                        })
                            .then(function (subscription) {
                                console.log('User subscribed successfully:', subscription.toJSON());
                                
                                // Send subscription to server
                                fetch('/subscribe', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ subscription })
                                })
                                    .then(response => {
                                        console.log('Server response:', response);
                                        if (response.ok) {
                                            console.log('Subscription saved on server');
                                        } else {
                                            console.error('Failed to save subscription on server:', response.status, response.statusText);
                                        }
                                    })
                                    .catch(function (fetchError) {
                                        console.error('Error during fetch to server:', fetchError);
                                    });
                            })
                            .catch(function (subscribeError) {
                                console.error('Failed to subscribe user:', subscribeError);
                            });
                    })
                    .catch(function (subscriptionError) {
                        console.error('Error checking existing subscription:', subscriptionError);
                    });
            })
            .catch(function (swReadyError) {
                console.error('Service Worker not ready:', swReadyError);
            });
    } else {
        console.error('Service Worker or PushManager not supported in this browser');
    }
}

// Helper function to parse Base64 keys
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding)
        .replace(/-/g, '+')
        .replace(/_/g, '/');

    const rawData = atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

// Call the function to subscribe when the page loads
document.addEventListener('DOMContentLoaded', function () {
    subscribeUser();
});
