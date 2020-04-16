USE ReallyARobot;

-- SELECT i.fullName, i.empId FROM employee_profiles p
-- INNER JOIN employee_images i
-- ON i.empId = p.empId;

-- USE information_schema;
-- SELECT *
-- FROM
--   KEY_COLUMN_USAGE
-- WHERE
--   REFERENCED_TABLE_NAME = 'employee_profiles'
--   AND REFERENCED_COLUMN_NAME = 'empId';

-- select referenced_column_name, table_name, column_name
-- from information_schema.KEY_COLUMN_USAGE
-- where table_schema = 'ReallyARobot'
-- and referenced_table_name = 'employee_profiles';
