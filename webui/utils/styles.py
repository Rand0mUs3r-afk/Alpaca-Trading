"""
Trading Agents Framework - CSS Styles
"""

# CSS for better styling
CSS = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}
.main-container {
    margin: 0;
    padding: 0;
}
.report-box {
    height: 500px;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    background-color: #f9f9f9;
}
.status-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.status-table td, .status-table th {
    border: 1px solid #ccc;
    padding: 10px;
    transition: background-color 0.5s ease;
}
.status-table tr:nth-child(even) {
    background-color: #f2f2f2;
}
.status-table tr:nth-child(odd) {
    background-color: #ffffff;
}
/* Ensure text is visible on all rows regardless of background */
.status-table td {
    color: #333;
    font-weight: 500;
}
.status-table th {
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: left;
    background-color: #2C3E50;
    color: white;
    font-weight: bold;
}
.pending {
    color: #7F8C8D !important;
    font-weight: bold;
}
.in-progress {
    color: #2980B9 !important;
    font-weight: bold;
    animation: pulse-blue 2s infinite;
}
.completed {
    color: #27AE60 !important;
    font-weight: bold;
    animation: flash-green 1s 1;
}
@keyframes pulse-blue {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}
@keyframes flash-green {
    0% { background-color: rgba(39, 174, 96, 0.3); }
    100% { background-color: transparent; }
}
.tabs {
    margin-top: 20px;
}
.time-period-btn {
    margin: 5px;
    padding: 8px 16px !important;
    border-radius: 4px;
    font-weight: bold !important;
    border: 1px solid #ddd !important;
    background-color: #f8f9fa !important;
    color: #333 !important;
}
.time-period-btn.active {
    background-color: #2C3E50 !important;
    color: white !important;
    border-color: #2C3E50 !important;
}
.chart-controls {
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 4px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
    display: flex;
    justify-content: center;
}
.chart-controls-heading {
    margin: 0;
    padding: 10px;
    background-color: #2C3E50;
    color: white;
    border-radius: 4px 4px 0 0;
    font-weight: bold;
    font-size: 16px;
}
.stats-container {
    margin-top: 15px;
    padding: 12px;
    background-color: #2C3E50;
    border-radius: 5px;
    font-size: 15px;
    color: white !important;
    font-weight: bold;
    text-align: center;
}
.auto-refresh-indicator {
    display: inline-block;
    margin-left: 10px;
    padding: 3px 8px;
    background-color: #27AE60;
    border-radius: 3px;
    font-size: 12px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}
"""

# JavaScript for auto-refresh and UI improvements
AUTO_REFRESH_JS = """
<script>
// Configuration
const REFRESH_INTERVAL = 500; // 0.5 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 500; // 0.5 seconds

// Track last update time
let lastUpdateTime = Date.now();
let updateCount = 0;
let lastStatusHtml = '';

// SUPER AGGRESSIVE DOM INJECTION
// This function will directly modify the DOM to force updates
function forceStatusUpdate() {
    // Find the status table
    const statusPanel = document.getElementById('status-panel');
    if (!statusPanel) return false;
    
    const statusTable = statusPanel.querySelector('table');
    if (!statusTable) return false;
    
    // Get all rows in the table
    const rows = statusTable.querySelectorAll('tr');
    if (rows.length <= 1) return false; // Header only
    
    // Get the current time to show when we last updated
    const now = new Date();
    const timestamp = now.toLocaleTimeString();
    
    // Add a hidden timestamp element to force browser to recognize changes
    let timestampEl = statusTable.querySelector('.update-timestamp');
    if (!timestampEl) {
        timestampEl = document.createElement('div');
        timestampEl.className = 'update-timestamp';
        timestampEl.style.display = 'none';
        statusTable.appendChild(timestampEl);
    }
    timestampEl.textContent = now.toISOString();
    
    // Create a visual indicator that an update happened
    let updateIndicator = document.getElementById('update-indicator');
    if (!updateIndicator) {
        updateIndicator = document.createElement('div');
        updateIndicator.id = 'update-indicator';
        updateIndicator.style.position = 'fixed';
        updateIndicator.style.top = '10px';
        updateIndicator.style.right = '10px';
        updateIndicator.style.backgroundColor = 'rgba(0,0,0,0.7)';
        updateIndicator.style.color = 'white';
        updateIndicator.style.padding = '5px 10px';
        updateIndicator.style.borderRadius = '5px';
        updateIndicator.style.fontSize = '12px';
        updateIndicator.style.zIndex = '1000';
        document.body.appendChild(updateIndicator);
    }
    updateIndicator.textContent = 'Updated: ' + timestamp;
    updateIndicator.style.backgroundColor = '#4CAF50';
    setTimeout(() => {
        updateIndicator.style.backgroundColor = 'rgba(0,0,0,0.7)';
    }, 500);
    
    // Check for terminal output to find completed tasks
    const pageText = document.body.innerText;
    
    // Look for patterns like "- Market Analyst: completed" in terminal output
    // This regex will find all instances of agent status updates
    const statusRegex = /- ([^:]+): (completed|in_progress|pending)/g;
    const agentStatuses = {};
    
    let match;
    while ((match = statusRegex.exec(pageText)) !== null) {
        const agentName = match[1].trim();
        const status = match[2];
        // Store the latest status for each agent
        agentStatuses[agentName] = status;
    }
    
    // Update the status table based on what we found
    let updated = false;
    for (let i = 1; i < rows.length; i++) { // Skip header row
        const row = rows[i];
        const cells = row.querySelectorAll('td');
        
        if (cells.length >= 3) {
            const agentNameCell = cells[1];
            const statusCell = cells[2];
            
            if (agentNameCell && statusCell) {
                const agentName = agentNameCell.textContent.trim();
                
                // If we found a status for this agent, update it
                if (agentStatuses[agentName]) {
                    const status = agentStatuses[agentName];
                    
                    // Only update if status has changed
                    if (status === 'completed' && !statusCell.textContent.includes('COMPLETED')) {
                        statusCell.innerHTML = '✅ COMPLETED';
                        statusCell.style.color = 'green';
                        row.style.animation = 'highlight 1s';
                        updated = true;
                    } else if (status === 'in_progress' && !statusCell.textContent.includes('IN PROGRESS')) {
                        statusCell.innerHTML = '🔄 IN PROGRESS';
                        statusCell.style.color = 'blue';
                        statusCell.style.animation = 'pulse 1.5s infinite';
                        row.style.animation = 'highlight 1s';
                        updated = true;
                    }
                }
            }
        }
    }
    
    // Force a repaint of the table
    if (updated) {
        statusTable.style.opacity = '0.99';
        setTimeout(() => { statusTable.style.opacity = '1'; }, 10);
    }
    
    return updated;
}

// Set up auto-refresh interval
function setupAutoRefresh() {
    console.log("[JS DEBUG] Setting up aggressive auto-refresh");
    
    // Initial update
    forceStatusUpdate();
    
    // Set up interval for regular updates
    setInterval(() => {
        forceStatusUpdate();
    }, REFRESH_INTERVAL);
    
    // Add visual indicator that auto-refresh is active
    const statusIndicator = document.getElementById('auto-refresh-status');
    if (statusIndicator) {
        statusIndicator.textContent = 'Auto-refresh active';
    }
    
    // Also hook into the refresh button if it exists
    const refreshBtn = document.getElementById('refresh-status-btn');
    if (refreshBtn) {
        // Add our own click handler
        refreshBtn.addEventListener('click', () => {
            forceStatusUpdate();
        });
    }
    
    console.log("[JS DEBUG] Aggressive auto-refresh setup complete");
}

// Set up the auto-refresh when the page is loaded
window.addEventListener('load', () => {
    console.log("[JS DEBUG] Page loaded, setting up aggressive auto-refresh");
    setTimeout(setupAutoRefresh, 1000);
    
    // Add keyboard shortcut (Ctrl+R) to manually trigger refresh
    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' && e.ctrlKey) {
            e.preventDefault();
            forceStatusUpdate();
        }
    });
});
</script>
""" 