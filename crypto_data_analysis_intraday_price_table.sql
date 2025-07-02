CREATE TABLE `intraday_price` (
  `symbol` varchar(255) DEFAULT NULL,
  `close_time` datetime DEFAULT NULL,
  `close` float DEFAULT NULL,
  `time` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
