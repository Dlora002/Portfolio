SELECT *
FROM model


SELECT *
FROM Price

SELECT *
FROM seller


SELECT *
FROM specifications

---We are going to drop unnecesary columns---

ALTER TABLE price
DROP COLUMN F7,F8,F9

---Observing which brand and type of vehicle are more expensive---

SELECT make,vehicle_type,price
FROM model md
JOIN price pr
	ON md.id = pr.id
ORDER BY price DESC
;
---Best selling brands---

SELECT make, COUNT(make) AS Brand_Count
FROM model
GROUP BY make
ORDER BY Brand_Count DESC
;
SELECT make,model,COUNT(model) AS Cars_sold
FROM model
GROUP BY make,model
ORDER BY count(model) DESC
;
---More sold per year model---
SELECT year,COUNT(id) AS Cars_sold
FROM price
GROUP BY year
ORDER BY COUNT(id) DESC
;
---Average price for vehicle type and transmission

SELECT vehicle_type, transmission, count(transmission) AS Count_Car_Transmission,ROUND(AVG(price),0) AS Avg_Price
FROM model md
JOIN specifications sp
ON md.id = sp.id
JOIN price pr
ON md.id = pr.id
WHERE vehicle_type IS NOT NULL AND transmission IS NOT NULL
GROUP BY vehicle_type,transmission
ORDER BY vehicle_type
;
---Max, Min, and Average price per brand

SELECT make, MAX(price) AS Max_price, MIN(price)  AS Min_price, ROUND(AVG(price),0) AS Average_price
FROM model md
JOIN  price pr
ON md.id = pr.id
WHERE price <> 0
GROUP BY make
ORDER BY make 
;
---Classifying cars by price---

SELECT make,
SUM(CASE WHEN price >= 60000 THEN 1 ELSE 0  END) AS Expensive,
SUM(CASE WHEN price BETWEEN 20000 AND 60000 THEN 1 ELSE 0 END) AS Affordable,
SUM(CASE WHEN price < 20000 THEN 1 ELSE 0 END) AS Cheap 
FROM model md
JOIN price pr
ON md.id = pr.id
GROUP BY make
ORDER BY make
;

---Total Expensive, Affordable, and Cheap vehicles
WITH CTE_VEHICLE AS
(
	SELECT make,
	SUM(CASE WHEN price >= 60000 THEN 1 ELSE 0 END) AS Expensive,
	SUM(CASE WHEN price BETWEEN 20000 AND 60000 THEN 1 ELSE 0 END) AS Affordable,
	SUM(CASE WHEN price < 20000 THEN 1 ELSE 0 END) AS Cheap 
	FROM model md
	JOIN price pr
	ON md.id = pr.id
	GROUP BY make
)
SELECT sum(Expensive) AS Expensive_Vehicles,sum(Affordable) AS Affordable_Vehicles,sum(Cheap) AS Cheap_Vehicles
FROM CTE_VEHICLE
;

---Vehicles sold by State and Cities in Ontario

SELECT state, COUNT(id) AS Cars_sold
FROM seller
WHERE state IS NOT NULL
GROUP BY state
ORDER BY COUNT(id) DESC

SELECT state, city, COUNT(id) AS Cars_sold
FROM seller
WHERE state IS NOT NULL AND state = 'ON' AND city IS NOT NULL
GROUP BY state,city
ORDER BY COUNT(id) DESC
;


---Best selling brand by state and cities in Ontario

WITH CTE_state AS
(
SELECT state,make, COUNT(make) AS Cars_sold,RANK() OVER (PARTITION BY state ORDER BY COUNT(make) DESC) AS Rank_Brand
FROM seller sel
JOIN model md
ON sel.id = md.id
WHERE state IS NOT NULL
GROUP BY state,make
)
SELECT state,MAX(Cars_sold) AS Max_vehicle_sold,make AS Max_Brand_sold
FROM CTE_state
WHERE Rank_Brand = 1
GROUP BY state,make
ORDER BY state
;

WITH CTE_City AS
(
SELECT City,make, COUNT(make) AS Cars_sold, RANK() OVER (PARTITION BY city ORDER BY COUNT(make) DESC) AS Rank_Brand
FROM seller sel
JOIN model md
ON sel.id = md.id
WHERE state = 'ON' AND City IS NOT NULL 
GROUP BY city,make
)
SELECT city,MAX(Cars_sold) AS Max_vehicle_sold,make AS Max_Brand_sold
FROM CTE_City
WHERE Rank_Brand = 1
GROUP BY city,make
ORDER BY city
;

---Best seller in Canada and by State

SELECT seller_name, COUNT(seller_name) AS Cars_Sold
FROM seller
GROUP BY seller_name
ORDER BY COUNT(seller_name) DESC
;

WITH CTE_Seller AS
(
SELECT seller_name, state, COUNT(seller_name) AS Cars_Sold, RANK() OVER (PARTITION BY state ORDER BY COUNT(seller_name) DESC) AS Rank_seller
FROM seller
GROUP BY seller_name,state
)
SELECT state, MAX(Cars_Sold) as Cars_sold,seller_name
FROM CTE_Seller
WHERE Rank_seller = 1 AND state IS NOT NULL
GROUP BY state,seller_name
ORDER BY state 














