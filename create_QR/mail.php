<?php
require 'db/connect_sv.php';
require "vendor/autoload.php";

//Import PHPMailer classes into the global namespace
//These must be at the top of your script, not inside a function
use Endroid\QrCode\QrCode;
use Endroid\QrCode\Writer\PngWriter;

use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;

//required files
require 'phpmailer/src/Exception.php';
require 'phpmailer/src/PHPMailer.php';
require 'phpmailer/src/SMTP.php';

session_start(); // Start the session

if (!isset($_SESSION['count'])) {
    $_SESSION['count'] = 1; // Bắt đầu với số 1
}

//Create an instance; passing `true` enables exceptions
if (isset($_POST["btn-reg"])) {
    
    $fullname = $_POST['fullname'];
    $gender = $_POST['gender'];
    $email = $_POST['email'];
    // date_default_timezone_set('Asia/Ho_Chi_Minh');
    // $currentDateTime = date("Y-m-d H:i:s");
    
    $prefix = "EvenA-";
    $qrid = $prefix . $_SESSION['count']; // Gán giá trị cho $qrid
    
    $_SESSION['count']++; // Tăng giá trị của count sau mỗi lần gửi form

    if(!empty($fullname) && !empty($gender) && !empty($email)){
        // echo "<pre>";
        // print_r($_POST);
        $sql = "INSERT INTO `yourtablename` (`QRID`, `Fullname`, `Gender`, `Email`) VALUES ('$qrid', '$fullname', '$gender', '$email')";
        if($conn->query($sql) == TRUE){
            $qr_code = QrCode::create($qrid);  
            $writer = new PngWriter;
            $result = $writer->write($qr_code);
            $result ->savetoFile("qr_code.png");
        }else{
            echo "
            <script> 
            alert('Error');
            document.location.href = 'index.php';
            </script> 
            {$sql}"
            .$conn->error;
        }
    }
    
    $mail = new PHPMailer(true);

    //Server settings
    $mail->isSMTP();                              //Send using SMTP
    $mail->Host       = 'smtp.gmail.com';       //Set the SMTP server to send through
    $mail->SMTPAuth   = true;             //Enable SMTP authentication
    $mail->Username   = 'nguyenkiet18072002@gmail.com';   //SMTP write your email
    $mail->Password   = 'dwsirshqnprqqmjw';      //SMTP password
    $mail->SMTPSecure = 'ssl';            //Enable implicit SSL encryption
    $mail->Port       = 465;                                    

    //Recipients
    $mail->setFrom( $_POST["email"], "Organization"); // Sender Email and name
    $mail->addAddress($_POST["email"]);     //Add a recipient email  
    $mail->addReplyTo($_POST["email"], $_POST["fullname"]); // reply to sender email

    //Content
    $mail->isHTML(true);               //Set email format to HTML
    $mail->Subject = 'QR code to entrance';  // email subject headings
    $mail->Body    = 'This is QR code'; //email message
    $mail->addEmbeddedImage('D:\software\xampp\xampp\htdocs\mailer\qr_code.png', 'qr_code');  
    // Success sent message alert
    $mail->send();
    echo
    " 
    <script> 
     alert('Message was sent successfully!');
     document.location.href = 'index.php';
    </script>
    ";
}
?>