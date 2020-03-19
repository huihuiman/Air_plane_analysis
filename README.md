<h3>對航空業客戶數據進行分析探討:<br></h3>
<br>
引入客戶數據，對客戶進行分類。<br>
對不同的客戶類別進行特徵分析，比較不同類別客戶的客戶價值。<br>
對不同價值的客戶類別提供個性化服務，制定相應的營銷策略。<br>
<br>
相關欄位說明<br>
<img src="https://github.com/huihuiman/Air_plane_analysis/blob/master/air%E5%9C%96%E7%89%87/data1.png?raw=true"> 
<img src="https://github.com/huihuiman/Air_plane_analysis/blob/master/air%E5%9C%96%E7%89%87/data2.png?raw=true">
<br>
1.消除機票為空的數據<br>
2.只保留機票不為0，平均折扣率不為0，總飛行公里數大於0的記錄。<br>

構建特徵值:<br>
客戶關係長度為L,消費時間間隔為R,消費頻率F,飛行里程M,折扣係數平均值為C<br>

<ol>L：LOAD_TIME測試區間內的結束時間-FFP_DATE入會時間<br>
R：LAST_TO_END最後一次乘機時間至測試區間內的結束時長<br>
F：FLIGHT_COUNT測試區間內的飛行次數<br>
M：SEG_KM_SUM測試區間內的總飛行公里數<br>
C：avg_discount平均折扣率<br></ol>



數據標準化<br>

使用k-means構建模型<br>

最後得出下圖客戶分類與相對應的價值分析圖<br>
<img src="https://github.com/huihuiman/Air_plane_analysis/blob/master/air%E5%9C%96%E7%89%87/20200319142101.png?raw=true">
<img src="https://github.com/huihuiman/Air_plane_analysis/blob/master/air%E5%9C%96%E7%89%87/20200319142133.png?raw=true">
