package model;

import java.io.Serializable;

public class Mutter implements Serializable {
	private int id;           // 各つぶやきに付けられるID
	private String userName;  // ユーザーネーム
	private String text;      // つぶやき内容
	private String dtime;     // つぶやいた日時
	
	public Mutter() {
		super();
	}
	public Mutter(int id, String userName, String text, String dtime) {
		super();
		this.id = id;
		this.userName = userName;
		this.text = text;
		this.dtime = dtime;
	}
	
	public int getId() {
		return id;
	}
	public String getUserName() {
		return userName;
	}
	public String getText() {
		return text;
	}
	public String getDtime() {
		return dtime;
	}
}
