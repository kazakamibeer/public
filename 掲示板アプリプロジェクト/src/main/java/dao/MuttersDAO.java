package dao;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import model.Mutter;

public class MuttersDAO {
	// データベース接続に使用する情報
	private final String JDBC_URL = "jdbc:h2:tcp://localhost/~/forum";
	private final String DB_USER = "sa";
	private final String DB_PASS = "";
	
	public List<Mutter> findAll(){
		List<Mutter> mutterList = new ArrayList<>();
		
		// JDBCドライバを読み込む
		try {
			Class.forName("org.h2.Driver");
		} catch(ClassNotFoundException e) {
			throw new IllegalStateException("JDBCドライバを読み込めませんでした");
		}
		
		// データベースに接続
		try(Connection conn =
				DriverManager.getConnection(JDBC_URL, DB_USER, DB_PASS)){
			
			// SELECT文の準備
			String sql =
					"select * from mutters order by id desc";
			PreparedStatement pStmt = conn.prepareStatement(sql);
			
			// SELECTを実行
			ResultSet rs = pStmt.executeQuery();
			
			// SELECT文の結果をArrayListに格納
			while(rs.next()) {
				int id = rs.getInt("id");
				String userName = rs.getString("name");
				String text = rs.getString("text");
				String dtime = rs.getString("dtime");
				Mutter mutter = new Mutter(id, userName, text, dtime);
				mutterList.add(mutter);
			}
		} catch(SQLException e) {
			e.printStackTrace();
			return null;
		}
		
		
		return mutterList;
	}
	
	public boolean create(Mutter mutter) {
		// JDBCドライバを読み込む
		try {
			Class.forName("org.h2.Driver");
		} catch(ClassNotFoundException e) {
			throw new IllegalStateException("JDBCドライバを読み込めませんでした");
		}
		
		// データベースに接続
		try(Connection conn =
				DriverManager.getConnection(JDBC_URL, DB_USER, DB_PASS)){
		
			// INSERT文の準備（idは自動連番なので指定しない）
			String sql = "insert into mutters(name, text, dtime) values(?, ?, ?)";
			PreparedStatement pStmt = conn.prepareStatement(sql);
		
			// INSERT文中の「?」に使用する値を設定してSQL文を完成
			pStmt.setString(1, mutter.getUserName());
			pStmt.setString(2, mutter.getText());
			pStmt.setString(3, mutter.getDtime());
		
			// INSERT文を実行（resultには追加された行数が入る）
			int result = pStmt.executeUpdate();
			if(result != 1) {
				return false;
			}
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public boolean remove(int id) {
		// JDBCドライバを読み込む
		try {
			Class.forName("org.h2.Driver");
		} catch(ClassNotFoundException e) {
			throw new IllegalStateException("JDBCドライバを読み込めませんでした");
		}
		
		// データベースに接続
		try(Connection conn = DriverManager.getConnection(JDBC_URL, DB_USER, DB_PASS)){
			
			// DELETE文を準備
			String sql = "delete from mutters where id = " + id;
			PreparedStatement pStmt = conn.prepareStatement(sql);
			
			// DELETEを実行し、結果を取得
			int result = pStmt.executeUpdate();
			
			// 削除が実行できなかった場合
			if(result != 1) {
				return false;
			}
			
		} catch(SQLException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
}