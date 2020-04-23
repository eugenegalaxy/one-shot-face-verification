DROP DATABASE IF EXISTS ReallyARobot;
CREATE DATABASE ReallyARobot;

USE ReallyARobot;

/*Table structure for table employee_profiles */

DROP TABLE IF EXISTS employee_profiles;

CREATE TABLE employee_profiles (
  empId smallint AUTO_INCREMENT PRIMARY KEY,
  fullName varchar(150) NOT NULL,
  age smallint NOT NULL,
  nationality varchar(30) NOT NULL,
  languageCode char(2) NOT NULL,
  heightCm smallint DEFAULT NULL,
  weightKg smallint DEFAULT NULL,
  socialMediaLink varchar(100) DEFAULT NULL
) ENGINE=INNODB;


/*Table structure for table employee_images */

DROP TABLE IF EXISTS employee_images;

CREATE TABLE employee_images (
  imgId smallint AUTO_INCREMENT PRIMARY KEY,
  fullName varchar(150) NOT NULL,
  empImage longblob DEFAULT NULL,
  empId smallint,
  CONSTRAINT employeeImgFK
  FOREIGN KEY (empId) 
  REFERENCES employee_profiles (empId)
    ON DELETE SET NULL 
    ON UPDATE CASCADE
) ENGINE=INNODB;
