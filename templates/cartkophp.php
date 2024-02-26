
<!-- //table banako//not necessary for flask -->
<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "tastetales";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// SQL statement to create the "cart" table
$sql = "CREATE TABLE cart (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_url VARCHAR(255),
    product_name VARCHAR(255),
    quantity INT,
    price INT
)";

/

// Close the database connection
$conn->close();
?>
