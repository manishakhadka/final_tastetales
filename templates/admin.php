<?php
$servername = "localhost";
$username = "root";
$password = "";
$database = "tastetales";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Fetch data from the orders table
$sql = "SELECT * FROM orders";
$result = $conn->query($sql);
?>

<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }

        h1 {
            text-align: center;
            color: #333;
        }

        table {
            border-collapse: collapse;
            width: 80%;
            margin: 20px auto;
            background-color: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        a {
            display: inline-block;
            padding: 5px 10px;
            text-decoration: none;
            color: #333;
            background-color: #e6e6e6;
            border: 1px solid #ccc;
            border-radius: 3px;
            margin-right: 5px;
        }

        a:hover {
            background-color: #d9d9d9;
        }
    </style>

    <!-- Bootstrap CSS -->
	<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
	<link rel="stylesheet" type="text/css" href="font-awesome/css/all.min.css">
	<link rel="stylesheet" type="text/css" href="css/owl.carousel.css"/>
	<link rel="stylesheet" type="text/css" href="css/owl.theme.default.min.css"/>
	<link rel="stylesheet" type="text/css" href="css/custom.css"/>
	<link rel="stylesheet" type="text/css" href="css/responsive.css"/>

	<script src="js/custom.js"></script>

	<title>Admin Page</title>
	
  </head>
	<body>
    <h1 style="text-align: center;">Admin Page</h1>
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
											<a class="nav-link" href="contact.html">Contact Us</a>
											  
										</div>										
									</li>

                                    <li class="nav-item">
										<div class="dropdown header">
											<a class="nav-link" href="#">Drinks</a>
											  <div class="dropdown-menu">
												<a class="dropdown-item" href="softdrinks.html">Soft Drinks</a>



												<!-- <button onclick="performAgeDetection()">Go to Hard Drinks</button>  -->
												<a class="dropdown-item" href="harddrinks.html">Hard Drinks</a>
												<!-- <a class="dropdown-item" href="#" onclick="redirectToHardDrinks()">Hard Drinks</a> -->

												

												
											  </div>
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
<body>







<h1 style="text-align: center;">Placed Order</h1>

    <table border="1">
        <tr>
            <th>ID</th>
            <th>Social Title</th>
            <th>First Name</th>
            <th>Last Name</th>
            <th>Contact Number</th>
            <th>Email</th>
            <th>Shipping Address</th>
            <th>Payment Option</th>
            <th>Message</th>
            <th>Actions</th>
        </tr>

        <?php
        // Display data in a table
        while ($row = $result->fetch_assoc()) {
            echo "<tr>";
            echo "<td>{$row['id']}</td>";
            echo "<td>{$row['social_title']}</td>";
            echo "<td>{$row['first_name']}</td>";
            echo "<td>{$row['last_name']}</td>";
            echo "<td>{$row['contact_number']}</td>";
            echo "<td>{$row['email']}</td>";
            echo "<td>{$row['shipping_address']}</td>";
            echo "<td>{$row['payment_option']}</td>";
            echo "<td>{$row['message']}</td>";
            echo "<td>";
            // echo "<a href='vieworder.php?id={$row['id']}'>View</a> ";
            echo "<a href='updateorder.php?id={$row['id']}'>Edit</a> ";
            echo "<a href='deleteorder.php?id={$row['id']}'>Delete</a>";
            echo "</td>";
            echo "</tr>";
        }

        // Close the database connection
        $conn->close();
        ?>
    </table>
</body>

		
		</div>
		
		<a id="back-to-top" href="#" class="btn back-to-top" role="button"><i class="fa fa-chevron-up"></i></a>	
		
		
		<script type="text/javascript" src="js/jquery.min.js"></script>
		<script type="text/javascript" src="js/popper.min.js"></script>
		<script type="text/javascript" src="js/bootstrap.min.js"></script>
		<script type="text/javascript" src="js/owl.carousel.js"></script>

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
		</script>









<!--extra haleko for cartlai functional banauna kam gareko xaina-->
<script>
    function addToCart(product) {
        // Check if the cart exists in localStorage
        let cart = JSON.parse(localStorage.getItem('cart')) || [];

        // Add the selected product to the cart
        cart.push(product);

        // Save the updated cart back to localStorage
        localStorage.setItem('cart', JSON.stringify(cart));

        alert('Product added to cart!');


    }
</script>
<script>
	
	function goToCartPage() {
	// Add any additional logic here if needed
	window.location.href = "cartpage.html";
}
</script>

<!-- <script>
	function performAgeDetection() {
		// Make an AJAX request to the Flask server
		fetch('/detected_age')
			.then(response => response.json())
			.then(data => {
				// Redirect to the hard drinks section if age is more than 18
				if (data.age && data.age >= 18) {
					window.location.href = '/harddrinks';
				} else {
					window.location.href = '/underage';
				}
			})
			.catch(error => console.error('Error:', error));
	}
</script> -->
<script>
	function redirectToHardDrinks() {
		// Perform age detection
		fetch('/detect_age')
			.then(response => response.json())
			.then(data => {
				const detectedAge = data.age;

				// Check if the detected age is suitable for accessing hard drinks
				if (isSuitableAge(detectedAge)) {
					// Redirect to the harddrinks page
					window.location.href = '/harddrinks';
				} else {
					// Handle underage case (you can customize this part)
					window.location.href = '/underage';
				}
			})
			.catch(error => {
				console.error('Error fetching age:', error);
			});
	}

	function isSuitableAge(detectedAge) {
		// Modify this function to check if the detected age is suitable for accessing hard drinks
		// For example, you can compare the age with 18
		return detectedAge >= 18;
	}
</script>

</html>







