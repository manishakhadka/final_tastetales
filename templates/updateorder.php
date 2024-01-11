<?php
$servername = "localhost";
$username = "root";
$password = "";
$database = "tastetales";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if (isset($_GET['id'])) {
    $order_id = $_GET['id'];

    // Fetch existing order details
    $selectSql = "SELECT * FROM orders WHERE id = $order_id";
    $result = $conn->query($selectSql);

    if ($result->num_rows > 0) {
        $order = $result->fetch_assoc();
    } else {
        echo "Order not found.";
        exit();
    }

    // Handle form submission for order update
    if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['edit_order'])) {
        $socialTitle = $_POST['gender'];
        $newFirstName = $_POST['Fname'];
        $newLastName = $_POST['Lname'];
        $contactNumber = $_POST['contact'];
        $email = $_POST['email'];
        $shippingAddress = $_POST['address'];
        $paymentOption = $_POST['payment'];
        $message = $_POST['message'];

        $updateSql = "UPDATE orders SET 
            social_title = '$socialTitle', 
            first_name = '$newFirstName', 
            last_name = '$newLastName', 
            contact_number = '$contactNumber', 
            email = '$email', 
            shipping_address = '$shippingAddress', 
            payment_option = '$paymentOption', 
            message = '$message'
            WHERE id = $order_id";

        if ($conn->query($updateSql) === TRUE) {
            // Redirect back to admin page after editing
            header("Location: admin.php");
            exit();
        } else {
            echo "Error updating order: " . $conn->error;
        }
    }
} else {
    echo "No order ID provided.";
    exit();
}

// Close the database connection
$conn->close();
?>

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
	

	<title>Order</title>
	
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
										<a class="nav-link" href="contact.html">Contact Us</a>
										  
									</div>										
								</li>

								<li class="nav-item">
									<div class="dropdown header">
										<a class="nav-link" href="#">Drinks</a>
										  <div class="dropdown-menu">
											<a class="dropdown-item" href="softdrinks.html">Soft Drinks</a>
											<a class="dropdown-item" href="harddrinks.html">Hard Drinks</a>
											
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
     
     <h1 style="text-align: center;">Edit Order</h1>

    

     <form method="post" action="updateorder.php?id=<?php echo $order_id; ?>">
        <section id="order-wrap">
                <div class="container">
                    <div class="row">
                        <div class="col-md-8">
                            <div class="order-form">
                                <h4 class="order-form-heading">PERSONAL INFORMATION</h4>
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Social title</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                            <input type="radio" name="gender" value="Mr." /checked> Mr.
                                            <input type="radio" name="gender" value="Miss."> Mrs.
                                            <input type="text" name="gender" value="<?php echo $order['social_title']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">First Name</div>
                                        </div>    
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                           
                                            <input type="text" name="Fname" value="<?php echo $order['first_name']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Last Name</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                            
                                        <input type="text" name="Lname" value="<?php echo $order['last_name']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Contact No.</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                        <input type="text" name="contact" value="<?php echo $order['contact_number']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Email</div>
                                        </div> 
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                        <input type="email" name="email" value="<?php echo $order['email']; ?>"><br>

                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Shipping Address</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                        <input type="text" name="address" value="<?php echo $order['shipping_address']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Payment Option</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                        <input type="text" name="payment" value="<?php echo $order['payment_option']; ?>"><br>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="order-form-wrap">
                                            <div class="order-form-titile">Message Box</div>
                                        </div>
                                    </div>
                                    <div class="col-md-8">
                                        <div class="order-input-form">
                                        <textarea name="message" rows="5"><?php echo $order['message']; ?></textarea><br>
                                        </div>
                                    </div>
                                </div>
                                <div style="text-align: right;"><button type="submit" name="edit_order">Update Order</button></div>
                                <!-- <div style="text-align: right;"><button class="btn-order" >Update Order</button></div> -->
                                <!-- <a href="admin.php">Back to Admin Panel</a> -->
                            </div>
                        </div>
                        
                    </div>
                </div>
            </div>    
        </section>
    </form>
    
</body>
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
            $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
            e.target // newly activated tab
            e.relatedTarget // previous active tab
            })
        </script>
        
        
	
</html>
			
			

    

