<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>

<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>岡本サンプルアプリ</title>
</head>
<body>
	<h1>【掲示板つくってみた】</h1>
	<p>
		<c:out value="${loginUser.name }"/>さん、ログイン中　
		<a href="Logout">ログアウト</a>
	</p>
	<p><a href="Main">更新</a>
	<form action="Main" method="post">
		<input type="text" name="text">
		<input type="submit" value="つぶやく">
	</form>
	
<c:if test="${not empty errorMsg }">
	<p style="color: red"><c:out value="${errorMsg }" /></p>
</c:if>

<c:forEach var="mutter" items="${mutterList }">
	<p>[<c:out value="${mutter.dtime }" />]<br>
	   <c:out value="${mutter.userName }" />　→　
	   <c:out value="${mutter.text }" />　<a href="Main?ID=${mutter.id }">削除</a></p>
</c:forEach>

</body>
</html>