<?php
$host = "103.18.6.82";
$username = "pck2uro1ax0t_sub";
$password = "sub@Nakhoa_1010";
$dbname = "pck2uro1ax0t_sub";

$conn = new mysqli($host, $username, $password, $dbname);

if ($conn->connect_error){
    die("Ket noi khong thanh cong:". $conn->connect_error);
}

?>