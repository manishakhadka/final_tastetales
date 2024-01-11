<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "tastetales";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get the item ID from the POST request
$itemId = $_POST['itemId'];

// Delete the item from the database
$sql = "DELETE FROM cart WHERE id = $itemId";
$result = $conn->query($sql);

if ($result) {
    echo "Item deleted successfully";
} else {
    echo "Error deleting item: " . $conn->error;
}

$conn->close();
?>
