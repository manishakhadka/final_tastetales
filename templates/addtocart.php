<!--index bata yesma aaucha --> 
    <?php 
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $image_url = $_POST["image_url"];
    $product_name = $_POST["product_name"];
    $total_quantity = $_POST["total_quantity"];
    $total_price = $_POST["total_price"];

    // Assuming you have configured your database connection, include the connection code here
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "tastetales";

    $conn = new mysqli($servername, $username, $password, $dbname);

    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // Insert data into the "cart" table
    $sql = "INSERT INTO cart (image_url, product_name, quantity, price) VALUES ('$image_url', '$product_name', $total_quantity, $total_price)";

    if ($conn->query($sql) === TRUE) {
        header("Location: cartpage.php");
    }

    // Close the database connection
    $conn->close();
}
?>
