package servlet;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import model.GetMutterListLogic;
import model.Mutter;
import model.PostMutterLogic;
import model.RemoveMutterListLogic;
import model.User;

/**
 * Servlet implementation class Main
 */
@WebServlet("/Main")
public class Main extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public Main() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// つぶやきの削除ボタンが押されたときの処理
		if(request.getParameter("ID") != null) {
			int id = Integer.parseInt(request.getParameter("ID"));
			
			RemoveMutterListLogic removeMutterListLogic = new RemoveMutterListLogic();
			removeMutterListLogic.remove(id);
		}		
		
		// つぶやきリストを取得して、リクエストスコープに保存
		GetMutterListLogic getMutterListLogic = new GetMutterListLogic();
		List<Mutter> mutterList = getMutterListLogic.execute();
		request.setAttribute("mutterList", mutterList);
		
		// セッションスコープからユーザー情報を取得
		// ログインしているか確認も兼ねる
		HttpSession session = request.getSession();
		User loginUser = (User)session.getAttribute("loginUser");
		
		if(loginUser == null) { // ログインしていない場合
			// リダイレクト
			response.sendRedirect("index.jsp");
		} else { // ログイン済みならフォワード
			RequestDispatcher dispatcher =
					request.getRequestDispatcher("WEB-INF/jsp/main.jsp");
			dispatcher.forward(request, response);
		}
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// リクエストパラメータの取得
		request.setCharacterEncoding("utf-8");
		String text = request.getParameter("text");
		
		// バリデーション
		if(text != null && text.length() != 0) {
			
			// 現在の日時を取得
			LocalDateTime nowDate = LocalDateTime.now();
			DateTimeFormatter dtf =  // 表示形式を指定
						DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH時mm分ss秒");
			String dtime = dtf.format(nowDate);
						
			// セッションスコープからユーザー情報を取得
			HttpSession session = request.getSession();
			User loginUser = (User)session.getAttribute("loginUser");
			
			// つぶやきを作成してつぶやきリストに追加
			Mutter mutter = new Mutter(0,loginUser.getName(), text, dtime);  // idは使わないので0に設定
			PostMutterLogic postMutterLogic = new PostMutterLogic();
			postMutterLogic.execute(mutter);
		}
		
		else {
			// エラーメッセージをリクエストスコープに保存
			request.setAttribute("errorMsg", "つぶやきが入力されていません");
		}
		
		// つぶやきリストを取得して、リクエストスコープに保存
		GetMutterListLogic getMutterListLogic = new GetMutterListLogic();
		List<Mutter> mutterList = getMutterListLogic.execute();
		request.setAttribute("mutterList", mutterList);
		
		// メイン画面にフォワード
		RequestDispatcher dispatcher =
				request.getRequestDispatcher("WEB-INF/jsp/main.jsp");
		dispatcher.forward(request, response);
	}

}
