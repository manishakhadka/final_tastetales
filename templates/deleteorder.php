<?php
$servername = "localhost";
$username = "root";
$password = "";
$database = "tastetales";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$message = "";  // Variable to store the deletion status message

if (isset($_GET['id'])) {
    $order_id = $_GET['id'];

    // Delete the order from the database
    $deleteSql = "DELETE FROM orders WHERE id = $order_id";

    if ($conn->query($deleteSql) === TRUE) {
        $message = "Order deleted successfully!";
    } else {
        $message = "Error deleting order: " . $conn->error;
    }
} else {
    $message = "No order ID provided.";
    exit();
}

// Close the database connection
$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delete Order</title>
</head>
<body>
    <h2>Delete Order</h2>

    <p><?php echo $message; ?></p>

    <a href="admin.php">Back to Admin Panel</a>
</body>
</html>
