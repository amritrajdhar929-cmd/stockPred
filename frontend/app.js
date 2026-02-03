const API_BASE = 'http://localhost:8000';
let predictionChart = null;

// Dark mode functionality
function toggleDarkMode() {
    const body = document.body;
    const darkModeIcon = document.getElementById('darkModeIcon');
    
    if (!body || !darkModeIcon) {
        console.error('Dark mode toggle: Required elements not found');
        return;
    }
    
    body.classList.toggle('dark-mode');
    
    if (body.classList.contains('dark-mode')) {
        darkModeIcon.classList.remove('fa-moon');
        darkModeIcon.classList.add('fa-sun');
        localStorage.setItem('darkMode', 'true');
    } else {
        darkModeIcon.classList.remove('fa-sun');
        darkModeIcon.classList.add('fa-moon');
        localStorage.setItem('darkMode', 'false');
    }
    
    // Update chart colors if chart exists
    if (predictionChart) {
        updateChartColors();
    }
}

// Check for saved dark mode preference
function initDarkMode() {
    const savedDarkMode = localStorage.getItem('darkMode');
    const darkModeIcon = document.getElementById('darkModeIcon');
    
    if (savedDarkMode === 'true') {
        document.body.classList.add('dark-mode');
        darkModeIcon.classList.remove('fa-moon');
        darkModeIcon.classList.add('fa-sun');
    }
}

// Update chart colors based on dark mode
function updateChartColors() {
    if (!predictionChart) return;
    
    const isDarkMode = document.body.classList.contains('dark-mode');
    const textColor = isDarkMode ? '#d1d5db' : '#6b7280';
    const gridColor = isDarkMode ? '#374151' : '#f3f4f6';
    const borderColor = isDarkMode ? '#60a5fa' : '#3b82f6';
    const backgroundColor = isDarkMode ? 'rgba(96, 165, 250, 0.1)' : 'rgba(59, 130, 246, 0.1)';
    
    predictionChart.options.scales.y.ticks.color = textColor;
    predictionChart.options.scales.y.grid.color = gridColor;
    predictionChart.options.scales.x.ticks.color = textColor;
    predictionChart.data.datasets[0].borderColor = borderColor;
    predictionChart.data.datasets[0].backgroundColor = backgroundColor;
    predictionChart.data.datasets[0].pointBackgroundColor = borderColor;
    
    predictionChart.update();
}

// Load popular stocks on page load
document.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
    loadPopularStocks();
    setupSearchAutocomplete();
    updateLastUpdatedTime();
});

async function loadPopularStocks() {
    try {
        const response = await fetch(`${API_BASE}/stocks`);
        const data = await response.json();
        
        const popularStocksDiv = document.getElementById('popularStocks');
        popularStocksDiv.innerHTML = '';
        
        // Show first 12 stocks as popular
        data.stocks.slice(0, 12).forEach(stock => {
            const stockCard = document.createElement('div');
            stockCard.className = 'stock-card glass-card rounded-xl p-4 cursor-pointer text-center';
            stockCard.innerHTML = `
                <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <span class="text-white font-bold text-sm">${stock.symbol.substring(0, 2)}</span>
                </div>
                <p class="font-semibold text-sm text-gray-900">${stock.symbol}</p>
                <p class="text-xs text-gray-600 mt-1">${stock.name.length > 15 ? stock.name.substring(0, 15) + '...' : stock.name}</p>
            `;
            stockCard.onclick = () => searchSpecificStock(stock.symbol);
            popularStocksDiv.appendChild(stockCard);
        });
    } catch (error) {
        console.error('Error loading popular stocks:', error);
    }
}

async function setupSearchAutocomplete() {
    try {
        const response = await fetch(`${API_BASE}/stocks`);
        const data = await response.json();
        
        const searchInput = document.getElementById('stockSearch');
        const suggestionsDiv = document.getElementById('searchSuggestions');
        
        searchInput.addEventListener('input', function() {
            const query = this.value.toUpperCase();
            if (query.length < 1) {
                suggestionsDiv.innerHTML = '';
                return;
            }
            
            const matches = data.stocks.filter(stock => 
                stock.symbol.includes(query) || stock.name.toUpperCase().includes(query)
            ).slice(0, 8);
            
            suggestionsDiv.innerHTML = '';
            matches.forEach(stock => {
                const suggestion = document.createElement('div');
                suggestion.className = 'glass-card rounded-lg px-4 py-3 cursor-pointer hover-lift';
                suggestion.innerHTML = `
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                            <span class="text-white font-bold text-xs">${stock.symbol.substring(0, 2)}</span>
                        </div>
                        <div>
                            <span class="font-semibold text-gray-900">${stock.symbol}</span>
                            <span class="text-gray-600 text-sm ml-2">${stock.name}</span>
                        </div>
                    </div>
                `;
                suggestion.onclick = () => {
                    searchInput.value = stock.symbol;
                    searchSpecificStock(stock.symbol);
                    suggestionsDiv.innerHTML = '';
                };
                suggestionsDiv.appendChild(suggestion);
            });
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (!searchInput.contains(e.target) && !suggestionsDiv.contains(e.target)) {
                suggestionsDiv.innerHTML = '';
            }
        });
        
    } catch (error) {
        console.error('Error setting up autocomplete:', error);
    }
}

async function searchStock() {
    const searchInput = document.getElementById('stockSearch');
    const symbol = searchInput.value.trim().toUpperCase();
    
    if (!symbol) {
        showNotification('Please enter a stock symbol', 'error');
        return;
    }
    
    await searchSpecificStock(symbol);
}

async function searchSpecificStock(symbol) {
    try {
        showLoading(true);
        
        // Fetch stock data and predictions
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol })
        });
        const data = await response.json();
        
        if (response.ok) {
            displayStockData(data);
            displayPredictions(data);
            updatePredictionChart(data);
            showNotification(`Analysis complete for ${symbol}`, 'success');
        } else {
            showNotification(data.detail || 'Stock not found', 'error');
            hideStockInfo();
        }
        
    } catch (error) {
        console.error('Error searching stock:', error);
        showNotification('Error fetching stock data', 'error');
        hideStockInfo();
    } finally {
        showLoading(false);
    }
}

function displayStockData(data) {
    const stockInfo = document.getElementById('stockInfo');
    const stockName = document.getElementById('stockName');
    const stockSymbol = document.getElementById('stockSymbol');
    const currentPrice = document.getElementById('currentPrice');
    
    stockName.textContent = data.name;
    stockSymbol.textContent = data.symbol;
    currentPrice.textContent = `₹${data.current_price.toFixed(2)}`;
    
    stockInfo.classList.remove('hidden');
}

function displayPredictions(data) {
    const predictionsSection = document.getElementById('predictionsSection');
    const pred1Day = document.getElementById('pred1Day');
    const pred5Day = document.getElementById('pred5Day');
    const pred30Day = document.getElementById('pred30Day');
    
    pred1Day.textContent = `₹${data.predictions['1_day'].toFixed(2)}`;
    pred5Day.textContent = `₹${data.predictions['5_day'].toFixed(2)}`;
    pred30Day.textContent = `₹${data.predictions['30_day'].toFixed(2)}`;
    
    // Update confidence bars
    updateConfidenceBar('conf1DayBar', 'conf1DayText', data.confidence_scores['1_day']);
    updateConfidenceBar('conf5DayBar', 'conf5DayText', data.confidence_scores['5_day']);
    updateConfidenceBar('conf30DayBar', 'conf30DayText', data.confidence_scores['30_day']);
    
    // Update prediction colors based on change
    updatePredictionColors(data);
    
    predictionsSection.classList.remove('hidden');
}

function updateConfidenceBar(barId, textId, confidence) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(textId);
    
    if (bar && text) {
        bar.style.width = `${confidence}%`;
        text.textContent = `${confidence}%`;
        
        // Update bar color based on confidence level
        bar.className = 'h-full transition-all duration-500';
        if (confidence >= 85) {
            bar.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
        } else if (confidence >= 80) {
            bar.style.background = 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)';
        } else {
            bar.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
        }
    }
}

function updatePredictionColors(data) {
    const currentPrice = data.current_price;
    
    // 1-day prediction
    const pred1Day = data.predictions['1_day'];
    const pred1DayElement = document.getElementById('pred1Day');
    const change1Day = ((pred1Day - currentPrice) / currentPrice) * 100;
    pred1DayElement.className = `text-2xl font-bold ${change1Day >= 0 ? 'text-green-600' : 'text-red-600'}`;
    
    // 5-day prediction
    const pred5Day = data.predictions['5_day'];
    const pred5DayElement = document.getElementById('pred5Day');
    const change5Day = ((pred5Day - currentPrice) / currentPrice) * 100;
    pred5DayElement.className = `text-2xl font-bold ${change5Day >= 0 ? 'text-green-600' : 'text-red-600'}`;
    
    // 30-day prediction
    const pred30Day = data.predictions['30_day'];
    const pred30DayElement = document.getElementById('pred30Day');
    const change30Day = ((pred30Day - currentPrice) / currentPrice) * 100;
    pred30DayElement.className = `text-2xl font-bold ${change30Day >= 0 ? 'text-green-600' : 'text-red-600'}`;
}

function updatePredictionChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    const labels = ['Current', '1 Day', '5 Days', '30 Days'];
    const prices = [
        data.current_price,
        data.predictions['1_day'],
        data.predictions['5_day'],
        data.predictions['30_day']
    ];
    
    const isDarkMode = document.body.classList.contains('dark-mode');
    const textColor = isDarkMode ? '#d1d5db' : '#6b7280';
    const gridColor = isDarkMode ? '#374151' : '#f3f4f6';
    const borderColor = isDarkMode ? '#60a5fa' : '#3b82f6';
    const backgroundColor = isDarkMode ? 'rgba(96, 165, 250, 0.1)' : 'rgba(59, 130, 246, 0.1)';
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Stock Price Prediction',
                data: prices,
                borderColor: borderColor,
                backgroundColor: backgroundColor,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointBackgroundColor: borderColor,
                pointBorderColor: '#fff',
                pointBorderWidth: 3,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: borderColor,
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `₹${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '₹' + value.toFixed(0);
                        },
                        color: textColor,
                        font: {
                            family: 'Inter'
                        }
                    },
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        color: textColor,
                        font: {
                            family: 'Inter'
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function showLoading(show) {
    const loadingState = document.getElementById('loadingState');
    if (show) {
        loadingState.classList.remove('hidden');
        document.getElementById('stockInfo').classList.add('hidden');
        document.getElementById('predictionsSection').classList.add('hidden');
    } else {
        loadingState.classList.add('hidden');
    }
}

function hideStockInfo() {
    document.getElementById('stockInfo').classList.add('hidden');
    document.getElementById('predictionsSection').classList.add('hidden');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-4 rounded-xl shadow-lg z-50 transform transition-all duration-300 ${
        type === 'error' ? 'bg-red-50 border border-red-200 text-red-800' : 'bg-blue-50 border border-blue-200 text-blue-800'
    }`;
    notification.innerHTML = `
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-full flex items-center justify-center ${
                type === 'error' ? 'bg-red-100' : 'bg-blue-100'
            }">
                <i class="fas ${type === 'error' ? 'fa-exclamation-circle text-red-600' : 'fa-info-circle text-blue-600'}"></i>
            </div>
            <div>
                <p class="font-medium">${message}</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 4000);
}

function updateLastUpdatedTime() {
    const lastUpdatedElement = document.getElementById('lastUpdated');
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-IN', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    lastUpdatedElement.textContent = timeString;
    
    // Update every minute
    setInterval(updateLastUpdatedTime, 60000);
}
