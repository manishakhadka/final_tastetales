<?php

$servername = "localhost";
$username = "root";
$password = "";

// Create connection
$conn = new mysqli($servername, $username, $password);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Create database
$sqlCreateDB = "CREATE DATABASE IF NOT EXISTS tastetales";
if ($conn->query($sqlCreateDB) === TRUE) {
    echo "Database created successfully or already exists.<br>";
} else {
    echo "Error creating database: " . $conn->error . "<br>";
}

// Select the database
$conn->select_db("tastetales");

// Create table
$sqlCreateTable = "CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    social_title VARCHAR(10),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    contact_number VARCHAR(20),
    email VARCHAR(50),
    shipping_address VARCHAR(255),
    payment_option VARCHAR(50),
    message TEXT
)";





if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $socialTitle = $_POST['gender'];
    $firstName = $_POST['Fname'];
    $lastName = $_POST['Lname'];
    $contactNumber = $_POST['contact'];
    $email = $_POST['email'];
    $shippingAddress = $_POST['address'];
    $paymentOption = $_POST['payment'];
    $message = $_POST['message'];

    $sql = "INSERT INTO orders (social_title, first_name, last_name, contact_number, email, shipping_address, payment_option, message) 
            VALUES ('$socialTitle', '$firstName', '$lastName', '$contactNumber', '$email', '$shippingAddress', '$paymentOption', '$message')";

    if ($conn->query($sql) === TRUE) {
        echo "Order placed successfully!";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

// Close connection
$conn->close();

?>
