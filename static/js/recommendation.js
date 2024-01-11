function getNameRecommendations() {
    const productName = document.getElementById('name').value;
    fetch('/get_name_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: productName }),
    })
    .then(response => response.json())
    .then(data => {
        const recommendationsDiv = document.getElementById('name-recommendations');
        recommendationsDiv.innerHTML = '<h3>Name Recommendations:</h3>';
        data.forEach(product => {
            recommendationsDiv.innerHTML += `<p>${product}</p>`;
        });
    });
}

function getBrandRecommendations() {
    const brandName = document.getElementById('brand').value;
    fetch('/get_brand_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ brand: brandName }),
    })
    .then(response => response.json())
    .then(data => {
        const recommendationsDiv = document.getElementById('brand-recommendations');
        recommendationsDiv.innerHTML = '<h3>Brand Recommendations:</h3>';
        data.forEach(product => {
            recommendationsDiv.innerHTML += `<p>${product}</p>`;
        });
    });
}

function getPriceRecommendations() {
    const priceValue = document.getElementById('price').value;
    fetch('/get_price_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ price: priceValue }),
    })
    .then(response => response.json())
    .then(data => {
        const recommendationsDiv = document.getElementById('price-recommendations');
        recommendationsDiv.innerHTML = '<h3>Price Recommendations:</h3>';
        data.forEach(product => {
            recommendationsDiv.innerHTML += `<p>${product}</p>`;
        });
    });
}
