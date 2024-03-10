<!-- index-addtocart-ani yo -->

<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Bootstrap CSS -->
	<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
	<link rel="stylesheet" type="text/css" href="font-awesome/css/all.min.css">
	<link rel="stylesheet" type="text/css" href="css/owl.carousel.css"/>
	<link rel="stylesheet" type="text/css" href="css/owl.theme.default.min.css"/>
	<link rel="stylesheet" type="text/css" href="css/custom.css"/>
	<link rel="stylesheet" type="text/css" href="css/responsive.css"/>
	

</head>
<body>
	<div id="app">
		<header>
			<div id="logo-header">
				<div class="container">
					<div class="row">
						<div class="col-md-4"></div>
						<div class="col-md-4 text-center">
							<a class="navbar-brand" href="#">
								<img src="img/logo-removebg-preview.png" width="160"/>
							</a>
						</div>
						<div class="col-md-4 my-auto text-right">
							<div class="header-right ml-auto">
								<form id="search-form">
									<div class="search-box">
										<input type="text" name="" class="search-input" placeholder="Search...."/> 
										<button class="search-btn"><i class="fa fa-search"></i></button>
									</div>						        
								</form>
								<div class="dropdown">
									<a href="#" class="dropdown" data-toggle="dropdown">
										<i class="fa fa-user"></i>
									</a>
									<div class="dropdown-menu">
									  <a class="dropdown-item" href="login.html">Sign in</a>
									  <a class="dropdown-item" href="register.html">Register</a>
									</div>
								</div>
								
								</div>
							</div>
							
						</div>
					</div>
				</div>
			</div>






				<div id="main-menu">
				<div class="container">

					<nav class="navbar navbar-toggleable-md navbar-expand-lg">
						<button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarTogglerDemo01" aria-controls="navbarTogglerDemo01" aria-expanded="false" aria-label="Toggle navigation">
							<i class="fa fa-bars"></i>
						</button>			
						<div class="collapse navbar-collapse" id="navbarTogglerDemo01">

							<ul class="navbar-nav mx-auto">


								<li class="nav-item">
									<div class="dropdown header">

										<a class="nav-link" href="index.html">Home</a>
										  
									</div>										
								</li>
								<li class="nav-item">
									<div class="dropdown header">
										<a class="nav-link" href="{{ url_for('contact') }}">Contact Us</a>
										  
									</div>										
								</li>

								<li class="nav-item">
										<div class="dropdown header">
											<a class="nav-link" href="#">Drinks</a>
											  <div class="dropdown-menu">
												<a class="dropdown-item" href="{{ url_for('softdrinks') }}">Soft Drinks</a>



												
												<a href="{{ url_for('permission') }}" class="dropdown-item">Hard Drinks</a>

											  </div>
										</div>										
									</li>


									<li class="nav-item">
										<div class="dropdown header">
											<a class="nav-link" href="{{ url_for('recommendation') }}">Search</a>
											
											  
										</div>										
									</li>

									</div>										
								</li>
							</ul>
						</div>
					</nav>
				</div>
			</div>
		
		</header>
		<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "tastetales";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT * FROM cart";
$result = $conn->query($sql);

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Cart</title>
    <!-- Include your CSS and JS files here -->
</head>
<body>
<section id="cartpage">
    <div class="container">
        <div class="row">
            <div class="col-md-9">
                <div class="cart-title">
                    Your Cart
                </div>
                <ul id="cartItems" style="list-style: none; padding: 0;">

                    {% if cart_items %}
                        {% for row in cart_items %}
                            <li class="cart-list-item" id="cartItem_{{ row.id }}">
                                <div class="col-1">
                                    <div class="remove-btn">
                                        <button class="remove-item-btn" onclick="deleteCartItem({{ row.id }})">
                                            <i class="fa fa-times"></i>
                                        </button>
                                    </div>
                                </div>
                                <div class="col-2">
                                    <div class="text-center">
                                        <img class="cart-item-img" src="{{ row.image_url }}">
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="cart-product-title">{{ row.product_name }}</div>
                                    <div class="cart-product-id">#{{ row.id }}</div>
                                </div>
                                <div class="col-2">
                                    <div>
                                        <span class="d-lg-none">Rs.{{ row.price }}</span>
                                    </div>
                                    <div class="cart-qty">
                                        Qty: {{ row.quantity }}
                                    </div>
                                </div>
                                <div class="col-1 d-none d-lg-block">
                                    <div class="cart-price">
                                        Rs.<br>{{ row.price }}
                                    </div>
                                </div>
                            </li>
                        {% endfor %}
                    {% else %}
                        <li class="cart-list-item">Your cart is empty.</li>
                    {% endif %}

                

                </ul>
            </div>
            <div class="col-md-3">
                <div class="cart-options">
                    <div class="checkout-go">
                        <a class="checkout-btn" href="order.html">Go to Checkout</a>
                    </div>
                    <div class="continue-shopping">
                        <a class="continue-btn" href="index.html">Continue Shopping</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<script>
    function deleteCartItem(itemId) {
        if (confirm("Are you sure you want to delete this item?")) {
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        var listItem = document.getElementById('cartItem_' + itemId);
                        if (listItem) {
                            listItem.remove();
                        }
                    } else {
                        alert('Error deleting item: ' + xhr.statusText);
                    }
                }
            };

            xhr.open('POST', 'deleteCartItem.php', true);
            xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
            xhr.send('itemId=' + itemId);
        }
    }
</script>

<!-- Include your scripts here -->

</body>
</html>

<?php
$conn->close();
?>


			<footer>
				<div class="container">
					<div id="footer-top">
						<div class="row">
							<div class="col-md-4 first">
								<h4>Contact Details</h4>
								<hr style="margin: 10px 0; border-top: 3px solid #009a46; width: 80px;">
								<div><i class="fa fa-map-marker-alt"></i>Kathmandu, Nepal</div>
								<div><i class="fa fa-phone"></i> +9779812345678/ +9779801234567</div>
								<div><i class="fa fa-envelope"></i><a href="mailto:info@greeline.com">info@TasteTales.com</a></div>
							</div>
							<div class="col-md-4 second">
								<h4>Quick Links</h4>
								<hr style="margin: 10px 0; border-top: 3px solid #009a46; width: 80px;">
								<p><a href="#">Home</a></p>
								<p><a href="#">About Us</a></p>
								<p><a href="#">Contact Us</a></p>
							</div>
							<div class="col-md-4 third">
								<h4>Useful Links</h4>
								<hr style="margin: 10px 0; border-top: 3px solid #009a46; width: 80px;">
								<div class="social">
									<p><a href="#">Offers</a></p>
									<p><a href="#">Terms & Conditions</a></p>
									<p><a href="#">Return Policy</a></p>
									<p><a href="#">Privacy Policy</a></p>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div id="footer-bottom">
					<div class="container">
						<div class="footer-social-section">
							<div class="footer-social">
								<div class="footer-social-text">
									Find us on Social Media
								</div>
								<div class="footer-social-icons">
									<a href="#" target="_blank">
										<i class="fab fa-facebook-f"></i>
									</a>
									<a href="#" target="_blank">
										<i class="fab fa-linkedin-in"></i>
									</a>
									<a href="#" target="_blank">
										<i class="fab fa-twitter"></i>
									</a>
									<a href="#" target="_blank">
										<i class="fab fa-instagram"></i>
									</a>
									<a href="#" target="_blank">
										<i class="fab fa-youtube"></i>
									</a>	
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-md-3">
								<div class="payments-footer">
									<img src="img/payment.png"/>
								</div>
							</div>
							<div class="col-md-6">
								<div class="copyright">© 2023 Taste Tales Pvt. Ltd. All Rights Reserved.
								</div>
							</div>
							
						</div>
					</div>
				</div>
			</footer>
		
		</div>
		
		<a id="back-to-top" href="#" class="btn back-to-top" role="button"><i class="fa fa-chevron-up"></i></a>	
		
		
		<script type="text/javascript" src="{{ url_for('static', filename='js/jquery.min.js') }}"></script>
<script type="text/javascript" src="{{ url_for('static', filename='js/popper.min.js') }}"></script>
<script type="text/javascript" src="{{ url_for('static', filename='js/bootstrap.min.js') }}"></script>
<script type="text/javascript" src="{{ url_for('static', filename='js/owl.carousel.js') }}"></script>



		<script>
			$(document).ready(function(){
				$(window).scroll(function () {
				if ($(this).scrollTop() > 50) {
					$('#back-to-top').fadeIn();
				} else {
					$('#back-to-top').fadeOut();
				}
			});
			// scroll body to 0px on click
			$('#back-to-top').click(function () {
				$('body,html').animate({
					scrollTop: 0
				}, 400);
				return false;
			});
			});
		</script>
		
		<script type="text/javascript">
			$('.dropdown-menu a.dropdown-toggle').on('click', function(e) {
            if (!$(this).next().hasClass('show')) {
                $(this).parents('.dropdown-menu').first().find('.show').removeClass("show");
            }
            var $subMenu = $(this).next(".dropdown-menu");
            $subMenu.toggleClass('show');

            $(this).parents('li.nav-item.dropdown.show').on('hidden.bs.dropdown', function(e) {
                $('.dropdown-submenu .show').removeClass("show");
            });

            return false;
        });
		</script>

		<script>
			$('#home-slide').owlCarousel({
				loop:true,
				autoplay:true,
				autoplayTimeout:3000,
				autoplayHoverPause:true,
				nav:true,
				navText : ["<i class='fa fa-chevron-left'></i>","<i class='fa fa-chevron-right'></i>"],
				responsive:{
					0:{
						items:1
					},
					600:{
						items:1
					},
					1000:{
						items:1
					}
				}
			})
		
			$('#recent-slide').owlCarousel({
				nav:true,
				margin:15,
				slideBy:'page',
				navText : ["<i class='fa fa-arrow-circle-left'></i>","<i class='fa fa-arrow-circle-right'></i>"],
				responsive:{
					0:{
						items:1
					},
					600:{
						items:2
					},
					1000:{
						items:4
					}
				}
			})

			$('#testimonialSlider').owlCarousel({
				nav:true,
				dots:false,
				margin:15,
				slideBy:'page',
				navText : ["<i class='fa fa-arrow-circle-left'></i>","<i class='fa fa-arrow-circle-right'></i>"],
				responsive:{
					0:{
						items:1
					},
					600:{
						items:2
					},
					1000:{
						items:3
					}
				}
			})

				$('#related-slide').owlCarousel({
				nav:true,
				margin:15,
				dots:false,
				slideBy:'page',
				navText : ["<i class='fa fa-arrow-circle-left'></i>","<i class='fa fa-arrow-circle-right'></i>"],
				responsive:{
					0:{
						items:1
					},
					600:{
						items:2
					},
					1000:{
						items:4
					}
				}
			})
		</script>
		<!--cartlai functional banaunalai indexko--kam garirako xaina-->
<script>
    // Load the cart items from localStorage
    let cart = JSON.parse(localStorage.getItem('cart')) || [];
    let cartItemsDiv = document.getElementById('cart-items');

    // Check if the cartItemsDiv exists before trying to append child elements
    if (cartItemsDiv) {
        // Display the cart items on the cart page
        cart.forEach(item => {
            let itemDiv = document.createElement('div');
            itemDiv.textContent = item;
            cartItemsDiv.appendChild(itemDiv);
        });
    } else {
        console.error("Element with id 'cart-items' not found.");
    }
</script>










    <script>
        // Retrieve the cart from local storage
        let cart = JSON.parse(localStorage.getItem('cart')) || [];

        // Display cart items on the cart page
        let cartItemsList = document.getElementById('cartItems');
        cart.forEach(item => {
            let listItem = document.createElement('li');
            listItem.textContent = `${item.productName} - Ksh.${item.price}`;
            cartItemsList.appendChild(listItem);
        });
    </script>







	</body>
</html>